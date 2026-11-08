"""Tests for the scrcpy bootstrap failure paths that stop a real device from playing.

The symptom these cover is the one that matters in practice::

    ADBError: scrcpy did not send its handshake byte. The bundled server jar / adb pair is
    probably mismatched.

That message used to be the end of the story — the server's own output was discarded and the run
aborted, even though ``screencap`` capture works fine on the same device. These tests pin down the
behaviour we want instead: read the server's error, explain it, clean up stale state, retry once,
and degrade to a backend that does work.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import List

import pytest

from cr_bot.adb.adb_utils import ADBError
from cr_bot.adb.scrcpy_multiclient import (
    MultiClient,
    ScreencapSession,
    ScrcpySession,
    diagnose_server_output,
)

SPRITES_DIR = pathlib.Path(__file__).resolve().parents[1] / "assets" / "unit_sprites"


class FakeAdb:
    """Records what the bot asks the device to do; answers the handful of calls we assert on."""

    def __init__(self, ps_output: str = "", jar_size: int | None = None,
                 forward_list: str = "") -> None:
        self.serial = "FAKE123"
        self.commands: List[List[str]] = []
        self._ps_output = ps_output
        self._jar_size = jar_size
        self._forward_list = forward_list
        self.pushes: List[str] = []

    # -- Adb surface used by ScrcpySession --------------------------------------------
    def base_cmd(self, include_serial: bool = True) -> List[str]:
        return ["adb"] + (["-s", self.serial] if include_serial else [])

    def connect(self, serial=None) -> str:
        return self.serial

    def display_info(self):
        return subprocess.CompletedProcess([], 0, b"", b"") if self._jar_size is None else self._info()

    def _info(self):
        from cr_bot.adb.adb_utils import DisplayInfo

        return DisplayInfo(width=1080, height=2400, rotation=0, density=450)

    def push(self, local, remote_dir="/data/local/tmp/", timeout: float = 60.0) -> None:
        self.pushes.append(str(local))

    def run(self, args, timeout: float = 10.0, include_serial: bool = True, check: bool = False):
        self.commands.append(list(args))
        joined = " ".join(args)
        if joined.startswith("shell ps"):
            return subprocess.CompletedProcess(args, 0, self._ps_output.encode(), b"")
        if joined.startswith("shell stat -c %s"):
            size = self._jar_size if self._jar_size is not None else -1
            return subprocess.CompletedProcess(args, 0, f"{size}\n".encode(), b"")
        if "forward --list" in joined:
            return subprocess.CompletedProcess(args, 0, self._forward_list.encode(), b"")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    # -- convenience ------------------------------------------------------------------
    def shells(self) -> List[str]:
        return [" ".join(c) for c in self.commands if c and c[0] == "shell"]


# ======================================================================================
# diagnosis
# ======================================================================================

def test_diagnosis_identifies_a_stale_server_holding_the_socket() -> None:
    text = "[server] ERROR: Could not listen on local socket scrcpy"
    advice = diagnose_server_output(text)
    assert "already running" in advice.lower()
    assert "pkill" in advice.lower()


def test_diagnosis_identifies_a_missing_or_truncated_jar() -> None:
    text = "java.lang.ClassNotFoundException: com.genymobile.scrcpy.Server"
    advice = diagnose_server_output(text)
    assert "jar" in advice.lower() and "push" in advice.lower()


def test_diagnosis_identifies_codec_rejection_and_suggests_knobs() -> None:
    text = "java.lang.IllegalStateException: Failed to initialize MediaCodec encoder"
    advice = diagnose_server_output(text)
    assert "bitrate" in advice.lower() or "fps" in advice.lower()


def test_diagnosis_is_empty_for_unrecognised_output() -> None:
    assert diagnose_server_output("") == ""
    assert diagnose_server_output("INFO: Device: Redmi") == ""


# ======================================================================================
# device housekeeping
# ======================================================================================

def test_stale_servers_are_killed_before_launching() -> None:
    ps = (
        "USER  PID  NAME\n"
        "shell 4242 com.genymobile.scrcpy.Server\n"
        "shell 4310 com.android.settings\n"
        "shell 5150 com.genymobile.scrcpy.Server\n"
    )
    adb = FakeAdb(ps_output=ps)
    session = ScrcpySession(adb)
    killed = session._kill_stale_servers()
    assert killed == 2
    shells = " | ".join(adb.shells())
    assert "kill -9 4242" in shells
    assert "kill -9 5150" in shells
    assert "kill -9 4310" not in shells, "unrelated processes must never be touched"


def test_stale_forward_to_the_scrcpy_socket_is_removed() -> None:
    adb = FakeAdb(forward_list="FAKE123 tcp:27183 localabstract:scrcpy\n"
                              "FAKE123 tcp:12345 localabstract:other\n")
    session = ScrcpySession(adb)
    session._kill_stale_servers()
    removals = [" ".join(c) for c in adb.commands if "forward --remove" in " ".join(c)]
    assert removals == ["forward --remove tcp:27183"]


def test_truncated_jar_push_is_reported_clearly(tmp_path) -> None:
    local = tmp_path / "scrcpy-server-v1.24.jar"
    local.write_bytes(b"x" * 500)
    adb = FakeAdb(jar_size=120)          # device only got 120 of 500 bytes
    session = ScrcpySession(adb)
    with pytest.raises(ADBError) as excinfo:
        session._verify_pushed_jar(local, "/data/local/tmp/scrcpy-server-v1.24.jar")
    message = str(excinfo.value)
    assert "120" in message and "500" in message
    assert "adb push" in message


def test_jar_check_is_silent_when_stat_is_unavailable(tmp_path) -> None:
    local = tmp_path / "scrcpy-server-v1.24.jar"
    local.write_bytes(b"x" * 500)
    adb = FakeAdb()                       # stat returns -1
    session = ScrcpySession(adb)
    session._verify_pushed_jar(local, "/data/local/tmp/scrcpy-server-v1.24.jar")


# ======================================================================================
# failure reporting
# ======================================================================================

def test_failure_report_includes_the_server_log_and_an_action() -> None:
    adb = FakeAdb()
    session = ScrcpySession(adb)
    session._server_log.append("[server] ERROR: Could not listen on local socket scrcpy")
    session._server_log.append("[server] java.io.IOException: bind failed")
    report = session.failure_report()
    assert "Could not listen" in report, "the raw server output must be preserved"
    assert "already running" in report.lower(), "and explained in plain language"


def test_failure_report_says_so_when_the_server_never_spoke() -> None:
    session = ScrcpySession(FakeAdb())
    report = session.failure_report()
    assert "no output at all" in report.lower()
    assert "/data/local/tmp" in report


# ======================================================================================
# graceful degradation
# ======================================================================================

class _ExplodingScrcpy(ScrcpySession):
    def start(self):
        raise ADBError(
            "scrcpy did not send its handshake byte.\n"
            "[server] ERROR: Could not listen on local socket scrcpy"
        )


class _RecordingScreencap(ScreencapSession):
    def start(self):
        self.serial = "FAKE123"
        return self


def test_auto_backend_degrades_to_screencap_instead_of_aborting(monkeypatch) -> None:
    """A broken scrcpy must not stop the bot: capture is a means, playing is the goal."""
    monkeypatch.setattr(
        "cr_bot.adb.scrcpy_multiclient.ScrcpySession", _ExplodingScrcpy
    )
    monkeypatch.setattr(
        "cr_bot.adb.scrcpy_multiclient.ScreencapSession", _RecordingScreencap
    )
    client = MultiClient(adb_path="adb", serial="FAKE123", backend="auto")
    client.start()
    try:
        assert client.backend_used == "screencap"
        assert "handshake" in client.fallback_reason
        assert isinstance(client.session(), _RecordingScreencap)
    finally:
        client.stop()


def test_explicit_scrcpy_backend_still_raises(monkeypatch) -> None:
    """If the operator asked for scrcpy specifically, do not silently downgrade."""
    monkeypatch.setattr("cr_bot.adb.scrcpy_multiclient.ScrcpySession", _ExplodingScrcpy)
    client = MultiClient(adb_path="adb", serial="FAKE123", backend="scrcpy")
    with pytest.raises(ADBError):
        client.start()


def test_screencap_backend_is_not_downgraded_again(monkeypatch) -> None:
    monkeypatch.setattr(
        "cr_bot.adb.scrcpy_multiclient.ScreencapSession",
        lambda adb, target_width=720: (_ for _ in ()).throw(ADBError("no screencap either")),
    )
    client = MultiClient(adb_path="adb", serial="FAKE123", backend="screencap")
    with pytest.raises(ADBError):
        client.start()


# ======================================================================================
# --stats must survive a broken link
# ======================================================================================

class _FakeStatsAdb:
    """Stands in for ``Adb`` inside ``--stats``: a healthy device with the game in front."""

    def __init__(self, *args, **kwargs) -> None:
        self.serial = "FAKE123"

    def connect(self, serial=None) -> str:
        return "FAKE123"

    def display_info(self):
        from cr_bot.adb.adb_utils import DisplayInfo

        return DisplayInfo(width=1080, height=2400, rotation=0, density=450)

    def foreground_package(self) -> str:
        return "com.supercell.clashroyale"

    def is_clash_royale_foreground(self, packages=None, hints=None) -> bool:
        from cr_bot.adb.adb_utils import is_game_package

        return is_game_package(self.foreground_package(), packages, hints)

    def find_game_package(self, packages=None, hints=None) -> str:
        return "com.supercell.clashroyale"

    def shell(self, command, timeout: float = 10.0) -> str:
        return ""


def test_stats_reports_a_dead_device_without_a_traceback(monkeypatch, capsys) -> None:
    from cr_bot.config.loader import BotConfig
    import main as cli

    class _NoDevice(_FakeStatsAdb):
        def connect(self, serial=None):
            raise ADBError("No Android device found over ADB")

    monkeypatch.setattr("cr_bot.adb.adb_utils.Adb", _NoDevice)
    code = cli.print_stats(BotConfig())
    out = capsys.readouterr().out
    assert code == 2
    assert "ADB DEVICE PROBLEM" in out
    assert "No Android device found" in out


def test_stats_reports_capture_failure_with_the_next_step(monkeypatch, capsys) -> None:
    from cr_bot.config.loader import BotConfig
    import main as cli

    class _DeadCapture(MultiClient):
        def start(self):
            raise ADBError("scrcpy did not send its handshake byte.\n[server] Could not listen")

    monkeypatch.setattr("cr_bot.adb.adb_utils.Adb", _FakeStatsAdb)
    monkeypatch.setattr("cr_bot.adb.scrcpy_multiclient.MultiClient", _DeadCapture)
    code = cli.print_stats(BotConfig())
    out = capsys.readouterr().out
    assert code == 2
    assert "CAPTURE FAILED" in out
    assert "--capture screencap" in out


# ======================================================================================
# foreground detection must work on modern Android
# ======================================================================================

@pytest.mark.parametrize(
    "dump",
    [
        "  topResumedActivity=ActivityRecord{2c1 u0 com.supercell.clashroyale/com.supercell.titan.GameApp t12}",
        "    mResumedActivity: ActivityRecord{abc u0 com.supercell.clashroyale/com.supercell.titan.GameApp t1}",
        "  mCurrentFocus=Window{9d2 u0 com.supercell.clashroyale/com.supercell.titan.GameApp}",
    ],
)
def test_foreground_package_parses_every_android_dump_format(dump: str) -> None:
    from cr_bot.adb.adb_utils import Adb

    class _Dump(Adb):
        def __init__(self) -> None:      # no ADB binary needed
            self.adb_bin = "adb"
            self.serial = "FAKE"

        def shell(self, command, timeout: float = 10.0) -> str:
            # only the probe that matches this Android build returns anything
            keyword = dump.strip().split("=")[0].split(":")[0].strip()
            return dump if keyword in command else ""

    assert _Dump().foreground_package() == "com.supercell.clashroyale"


def test_foreground_package_returns_empty_when_nothing_matches() -> None:
    from cr_bot.adb.adb_utils import Adb

    class _Dump(Adb):
        def __init__(self) -> None:
            self.adb_bin = "adb"
            self.serial = "FAKE"

        def shell(self, command, timeout: float = 10.0) -> str:
            return ""

    assert _Dump().foreground_package() == ""


# ======================================================================================
# the retry ladder must speak the same protocol dialect as the jar it launches
# ======================================================================================

def test_the_ladder_uses_the_real_jar_version_not_a_hardcoded_one() -> None:
    """The bug that made scrcpy unusable on every device.

    ``jar_version`` defaulted to (1, 24, 0) until ``_launch_server`` parsed the jar — but the
    ladder was planned *before* that, from the default. A v5 jar was therefore driven with
    legacy option names, the server answered ``Unknown server option: bit_rate``, and every
    attempt after the first was guaranteed to fail.
    """
    from cr_bot.adb.scrcpy_multiclient import ScrcpySession

    session = ScrcpySession.__new__(ScrcpySession)
    session.jar_path = None
    session.bitrate = 12_000_000
    session._resolve_jar()

    assert session.jar_version[0] >= 2, "the bundled jar is modern; the test assumes that"
    legacy_names = {"bit_rate", "encoder_name", "lock_video_orientation"}
    for label, options in session._attempts():
        clash = legacy_names & set(options)
        assert not clash, f"attempt {label!r} passes legacy option(s) {clash} to a modern server"


def test_a_legacy_jar_gets_legacy_options_only() -> None:
    from cr_bot.adb import scrcpy_protocol as proto
    from cr_bot.adb.scrcpy_multiclient import ScrcpySession

    session = ScrcpySession.__new__(ScrcpySession)
    session.jar_path = "scrcpy-server-v1.24.jar"
    session.bitrate = 12_000_000
    session._resolve_jar()
    assert session.jar_version == (1, 24, 0)
    modern_names = {"video_bit_rate", "video_encoder", "scid", "audio", "video"}
    for label, options in session._attempts():
        clash = modern_names & set(options)
        assert not clash, f"attempt {label!r} passes modern option(s) {clash} to a 1.x server"


def test_the_ladder_offers_a_smaller_stream_for_refusing_encoders() -> None:
    from cr_bot.adb.scrcpy_multiclient import ScrcpySession

    session = ScrcpySession.__new__(ScrcpySession)
    session.jar_path = None
    session.bitrate = 12_000_000
    session._resolve_jar()
    labels = [label for label, _ in session._attempts()]
    assert any("smaller" in label for label in labels), labels
    assert any("fresh socket" in label for label in labels), labels
