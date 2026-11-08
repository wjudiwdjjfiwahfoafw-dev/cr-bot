"""Custom builds of the game (Nulls Royale, private servers, modded APKs).

These are the *same game* with a different application id. Anything that hard-codes
``com.supercell.clashroyale`` therefore reports "the game is not running" while the user is
staring at a live battle - exactly what happened with ``nullsroyale.rel.free``.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from cr_bot.adb.adb_utils import (
    GAME_PACKAGE,
    Adb,
    is_game_package,
)
from cr_bot.config.loader import BotConfig, load_config

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ======================================================================================
# package matching
# ======================================================================================

@pytest.mark.parametrize(
    "package",
    [
        "com.supercell.clashroyale",
        "com.supercell.clashroyale.nulls",       # re-signed variant keeping the prefix
        "nullsroyale.rel.free",                  # the build that exposed the bug
        "nullsroyale.rel.paid",
        "com.nulls.clashroyale",                 # hint match
    ],
)
def test_known_builds_are_recognised(package: str) -> None:
    assert is_game_package(package) is True


@pytest.mark.parametrize(
    "package",
    [
        "",
        "com.android.settings",
        "com.supercell.brawlstars",              # same publisher, different game
        "org.nulls.royale.client",               # nothing in common with the hints
        "com.supercell.clashofclans",
    ],
)
def test_other_apps_are_not_mistaken_for_the_game(package: str) -> None:
    assert is_game_package(package) is False


def test_explicit_configuration_wins() -> None:
    assert is_game_package("org.nulls.royale.client") is False
    assert is_game_package(
        "org.nulls.royale.client", packages=["org.nulls.royale.client"]
    ) is True
    assert is_game_package("my.custom.build", hints=["custom"]) is True
    # an explicit list must not silently keep the built-in hints
    assert is_game_package("nullsroyale.rel.free", packages=["com.other.game"], hints=[]) is False


class _DumpAdb(Adb):
    """Adb whose shell prints canned output, so no device is needed."""

    def __init__(self, foreground: str = "", packages: str = "") -> None:
        self.adb_bin = "adb"
        self.serial = "FAKE"
        self._foreground = foreground
        self._packages = packages
        self.launched: list = []

    def shell(self, command, timeout: float = 10.0) -> str:
        if "pm list packages" in command:
            return self._packages
        if any(k in command for k in ("topResumedActivity", "mResumedActivity",
                                      "mCurrentFocus", "mFocusedApp")):
            return self._foreground
        if command.startswith("monkey"):
            self.launched.append(command)
            return "Events injected: 1"
        return ""


def test_foreground_check_accepts_the_custom_build() -> None:
    adb = _DumpAdb(foreground="  mCurrentFocus=Window{1 u0 nullsroyale.rel.free/nullsroyale.Main}")
    assert adb.is_clash_royale_foreground() is True
    assert adb.foreground_package() == "nullsroyale.rel.free"


def test_foreground_check_still_rejects_an_unrelated_app() -> None:
    adb = _DumpAdb(foreground="  mCurrentFocus=Window{1 u0 com.android.launcher3/.Launcher}")
    assert adb.is_clash_royale_foreground() is False


def test_installed_game_is_found_when_it_is_not_in_front() -> None:
    adb = _DumpAdb(
        foreground="  mCurrentFocus=Window{1 u0 com.android.launcher3/.Launcher}",
        packages=textwrap.dedent(
            """\
            package:com.android.settings
            package:nullsroyale.rel.free
            package:com.spotify.music
            """
        ),
    )
    assert adb.is_clash_royale_foreground() is False
    assert adb.find_game_package() == "nullsroyale.rel.free"


def test_find_game_package_returns_empty_when_nothing_matches() -> None:
    adb = _DumpAdb(packages="package:com.android.settings\npackage:com.spotify.music\n")
    assert adb.find_game_package() == ""


def test_launching_uses_the_package_id_not_a_hard_coded_one() -> None:
    adb = _DumpAdb()
    assert adb.launch_package("nullsroyale.rel.free") is True
    assert "nullsroyale.rel.free" in adb.launched[0]
    assert "android.intent.category.LAUNCHER" in adb.launched[0]


def test_launch_package_reports_failure_when_no_activity_exists() -> None:
    class _NoActivity(_DumpAdb):
        def shell(self, command, timeout: float = 10.0) -> str:
            return "monkey aborted: No activities found to run"

    assert _NoActivity().launch_package("com.nope") is False
    assert _DumpAdb().launch_package("") is False


# ======================================================================================
# configuration
# ======================================================================================

def test_config_ships_the_custom_build_by_default() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "mk_zap_bait.yaml")
    assert GAME_PACKAGE in cfg.game_packages
    assert "nullsroyale.rel.free" in cfg.game_packages
    assert any("nullsroyale" in hint for hint in cfg.game_package_hints)


def test_config_can_be_pointed_at_any_build(tmp_path) -> None:
    config = tmp_path / "bot.yaml"
    source = (REPO_ROOT / "configs" / "mk_zap_bait.yaml").read_text(encoding="utf-8")
    config.write_text(source + "\ngame:\n  packages: [\"org.custom.client\"]\n", encoding="utf-8")
    cfg = load_config(config)
    assert cfg.game_packages == ["org.custom.client"]


def test_defaults_exist_without_any_config() -> None:
    cfg = BotConfig()
    assert cfg.game_packages
    assert cfg.game_package_hints


def test_stats_names_the_installed_build(monkeypatch, capsys) -> None:
    """`--stats` must not say "no build of the game" when a custom APK is installed."""
    import main as cli

    class _CustomBuildAdb(_DumpAdb):
        def connect(self, serial=None) -> str:
            return "RZCT20A0MBJ"

        def display_info(self):
            from cr_bot.adb.adb_utils import DisplayInfo

            return DisplayInfo(width=1080, height=2400, rotation=0, density=450)

        def foreground_package(self) -> str:
            return "com.android.launcher3"

        def find_game_package(self, packages=None, hints=None) -> str:
            return "nullsroyale.rel.free"

    monkeypatch.setattr("cr_bot.adb.adb_utils.Adb", lambda *a, **k: _CustomBuildAdb())
    monkeypatch.setattr(
        "cr_bot.adb.scrcpy_multiclient.MultiClient.start",
        lambda self: (_ for _ in ()).throw(__import__("cr_bot.adb.adb_utils", fromlist=["ADBError"]).ADBError("boom")),
    )
    cli.print_stats(BotConfig())
    out = capsys.readouterr().out
    assert "nullsroyale.rel.free" in out
    assert "no build of the game found" not in out


# ======================================================================================
# scrcpy server selection
# ======================================================================================

def test_newest_bundled_server_is_preferred() -> None:
    from cr_bot.adb.scrcpy_multiclient import available_jars, find_scrcpy_jar

    jars = available_jars()
    assert jars, "at least one server jar must ship with the repo"
    from cr_bot.adb.scrcpy_protocol import parse_server_version

    newest = max(parse_server_version(p.name) for p in jars)
    assert parse_server_version(find_scrcpy_jar().name) == newest


def test_an_explicit_server_jar_wins(tmp_path) -> None:
    from cr_bot.adb.scrcpy_multiclient import find_scrcpy_jar

    custom = tmp_path / "scrcpy-server-v1.24.jar"
    custom.write_bytes(b"fake")
    assert find_scrcpy_jar(str(custom)) == custom


def test_missing_explicit_jar_is_reported(tmp_path) -> None:
    from cr_bot.adb.scrcpy_multiclient import find_scrcpy_jar

    with pytest.raises(FileNotFoundError):
        find_scrcpy_jar(str(tmp_path / "nope.jar"))


def test_the_bundled_modern_server_is_a_real_jar() -> None:
    """Guard against committing a placeholder: a jar starts with PK 0x03 0x04."""
    from cr_bot.adb.scrcpy_multiclient import available_jars

    modern = [p for p in available_jars() if "5.0.1" in p.name]
    assert modern, "the v5 server should be bundled"
    head = modern[0].read_bytes()[:4]
    assert head[:2] == b"PK", f"{modern[0].name} does not look like a jar: {head!r}"
