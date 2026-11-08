"""Opt-in deployment learning must not treat a nearby guess as ground truth."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cr_bot.bot_runner import ClashRoyaleBotRunner
from cr_bot.config.loader import BotConfig, load_config
from cr_bot.models import BattleState, BotAction, DetectedUnit, GamePhase, Side
from cr_bot.vision.unit_identity import UnitIdentityMemory


@pytest.fixture
def runner(tmp_path):
    config = BotConfig(identity_dir=str(tmp_path), enable_identity_learning=True,
                       enable_unit_recognition=False)
    obj = ClashRoyaleBotRunner(config, show_hud=False, auto_calibrate=False,
                               deploy_retries=0, manual_input=False)
    obj._was_in_battle = True
    return obj


def tracked(track_id=2, side=Side.ALLY, name="unknown_medium", source="segmentation",
            confidence=.85, hp=1.0, seen=10.0, box=(75, 75, 125, 125)):
    return DetectedUnit(track_id=track_id, name=name, side=side, tile_x=5, tile_y=20,
                        pixel_bbox=box, source=source, confidence=confidence,
                        hp_ratio=hp, first_seen_ts=seen, last_seen_ts=seen)


def queue(runner, card="mega_knight", excluded=frozenset({1})):
    return runner._queue_identity_sample(card, (100, 100), excluded, runner._match_epoch)


def new_frame():
    return np.random.default_rng(2).integers(0, 255, (200, 200, 3), dtype=np.uint8)


def test_learning_is_opt_in_and_old_memory_namespace_is_not_reused(tmp_path):
    assert BotConfig().enable_identity_learning is False
    settings = tmp_path / "config.yaml"
    settings.write_text("vision:\n  unit_recognition: false\n  identity_learning: true\n")
    cfg = load_config(settings)
    assert not cfg.enable_unit_recognition and cfg.enable_identity_learning
    cfg.identity_dir = str(tmp_path)
    bot = ClashRoyaleBotRunner(cfg, show_hud=False, auto_calibrate=False, manual_input=False)
    assert bot.arena_detector.cv_fallback.enable_recognition is False
    assert bot.identity.namespace.endswith("_guarded_v2")


def test_unique_new_clean_ally_must_be_seen_twice_before_persisting(runner):
    assert queue(runner)
    frame = new_frame()
    ally = tracked()
    runner._learn_from_frame(frame, [ally], 10.0)
    assert runner.identity.sample_count == 0
    runner._learn_from_frame(frame, [ally], 10.0)  # same observation cannot vote twice
    assert runner.identity.sample_count == 0
    ally.last_seen_ts = 10.1
    runner._learn_from_frame(frame, [ally], 10.1)
    assert runner.identity.sample_counts() == {"mega_knight": 1}
    assert runner.identity.path.is_file()
    reloaded = UnitIdentityMemory(runner.identity.directory, runner.identity.namespace)
    assert reloaded.sample_counts() == {"mega_knight": 1}
    assert not runner._learning_requests


@pytest.mark.parametrize("card", ["zap", "fireball", "cannon", "goblin_drill", "bats",
                                  "goblin_gang", "goblin_barrel", "archers",
                                  "royal_hogs", "elite_barbarians", "wall_breakers",
                                  "unknown_medium", "unlisted_card"])
def test_spells_buildings_groups_and_unknown_cards_never_queue(runner, card):
    assert not queue(runner, card)
    assert not runner._learning_requests


@pytest.mark.parametrize("case", ["enemy", "existing", "bars", "low_confidence", "damaged",
                                  "conflicting_name", "stale", "multiple"])
def test_ambiguous_or_unverified_scenes_are_not_learned(runner, case):
    assert queue(runner)
    item = tracked()
    if case == "enemy":
        item.side = Side.ENEMY
    elif case == "existing":
        item.track_id = 1
    elif case == "bars":
        item.source = "bars"
    elif case == "low_confidence":
        item.confidence = .3
    elif case == "damaged":
        item.hp_ratio = .5
    elif case == "conflicting_name":
        item.name = "giant"
    elif case == "stale":
        item.last_seen_ts = 8.0
    units = [item, tracked(track_id=3, side=Side.ENEMY)] if case == "multiple" else [item]
    frame = new_frame()
    runner._learn_from_frame(frame, units, 10.0)
    if case != "stale":
        for unit in units:
            unit.last_seen_ts = 10.1
    runner._learn_from_frame(frame, units, 10.1)
    assert runner.identity.sample_count == 0
    assert not runner.identity.path.exists()


def test_intervening_ambiguity_resets_two_frame_confirmation(runner):
    assert queue(runner)
    frame = new_frame()
    ally = tracked()
    runner._learn_from_frame(frame, [ally], 10.0)
    ally.last_seen_ts = 10.1
    enemy = tracked(track_id=3, side=Side.ENEMY, seen=10.1)
    runner._learn_from_frame(frame, [ally, enemy], 10.1)
    ally.last_seen_ts = 10.2
    runner._learn_from_frame(frame, [ally], 10.2)
    assert runner.identity.sample_count == 0
    ally.last_seen_ts = 10.3
    runner._learn_from_frame(frame, [ally], 10.3)
    assert runner.identity.sample_count == 1


def test_expired_and_previous_match_requests_are_discarded(runner):
    assert queue(runner)
    runner._learning_requests[0].expires_at = 0.0
    runner._learn_from_frame(new_frame(), [tracked()], 10.0)
    assert not runner._learning_requests
    old_epoch = runner._match_epoch
    assert queue(runner)
    runner.reset_match_state(11.0)
    assert not runner._learning_requests
    assert not runner._queue_identity_sample("mega_knight", (100, 100), frozenset(), old_epoch)
    assert runner.identity.sample_count == 0


def test_same_track_cannot_supervise_two_different_plays(runner):
    assert queue(runner, "mega_knight")
    assert queue(runner, "knight")
    frame, ally = new_frame(), tracked()
    runner._learn_from_frame(frame, [ally], 10.0)
    ally.last_seen_ts = 10.1
    runner._learn_from_frame(frame, [ally], 10.1)
    assert runner.identity.sample_count == 1
    assert runner.identity.known_cards == ["mega_knight"]


class FakeADB:
    input_backend = "fake"

    def __init__(self, ok=True):
        self.ok = ok

    def set_input_backend(self, backend):
        self.input_backend = backend

    def deploy_card(self, slot, arena):
        return SimpleNamespace(ok=self.ok, backend=self.input_backend, note="")


def test_verified_deploy_only_queues_learning_without_running_detector_on_worker(runner, monkeypatch):
    runner.adb = FakeADB()
    runner._hand_snapshot = lambda: object()
    runner._wait_for_hand_change = lambda *args: True
    monkeypatch.setattr(runner.arena_detector, "detect_and_track",
                        lambda *args: pytest.fail("worker must not run the stateful detector"))
    action = BotAction("mega_knight", 0, 9.0, 22.0, 1.0, "test")
    outcome = runner._execute_deploy(action, (50, 50), (100, 100), frozenset({1}),
                                     runner._match_epoch)
    assert outcome.ok
    assert len(runner._learning_requests) == 1
    assert runner.identity.sample_count == 0  # main vision thread samples later


def test_failed_input_cannot_be_verified_by_an_unrelated_hand_change(runner, monkeypatch):
    runner.adb = FakeADB(ok=False)
    runner._hand_snapshot = lambda: object()
    runner._wait_for_hand_change = lambda *args: pytest.fail("failed input should not be verified")
    action = BotAction("mega_knight", 0, 9.0, 22.0, 1.0, "test")
    outcome = runner._execute_deploy(action, (50, 50), (100, 100), frozenset(),
                                     runner._match_epoch)
    assert not outcome.ok
    assert not runner._learning_requests


def test_preexisting_nearby_unit_disables_learning_even_if_play_proceeds(runner, monkeypatch):
    monkeypatch.setattr(runner, "_start_deploy_worker", lambda: None)
    action = BotAction("mega_knight", 0, 9.0, 22.0, 1.0, "test")
    px, py = runner.grid.tile_to_pixel(action.tile_x, action.tile_y)
    existing = tracked(track_id=1, box=(px-15, py-20, px+15, py+20))
    state = BattleState(10.0, GamePhase.BATTLE_1X, 10.0, [], units=[existing])
    assert runner.submit_action(action, state)
    assert runner._pending_deploy[3] is None  # no safe pre-deploy snapshot for learning


def test_nearby_unit_without_box_still_makes_scene_ambiguous(runner):
    assert queue(runner)
    ally = tracked()
    enemy = tracked(track_id=3, side=Side.ENEMY, box=(0, 0, 0, 0))
    enemy.tile_x, enemy.tile_y = runner.grid.pixel_to_tile(100, 100)
    frame = new_frame()
    runner._learn_from_frame(frame, [ally, enemy], 10.0)
    ally.last_seen_ts = enemy.last_seen_ts = 10.1
    runner._learn_from_frame(frame, [ally, enemy], 10.1)
    assert runner.identity.sample_count == 0


def test_tentative_tracker_detections_count_in_pre_deploy_scene(runner, monkeypatch):
    from cr_bot.models import MoveType, WeightClass
    from cr_bot.vision.arena_detector import RawDetection

    monkeypatch.setattr(runner, "_start_deploy_worker", lambda: None)
    action = BotAction("mega_knight", 0, 9.0, 22.0, 1.0, "test")
    px, py = runner.grid.tile_to_pixel(action.tile_x, action.tile_y)
    bar = RawDetection("unknown_medium", Side.ENEMY, 9.0, 22.0,
                       (px-15, py-20, px+15, py+20), .45, 1.0,
                       MoveType.GROUND, WeightClass.MEDIUM, "bars")
    assert runner.arena_detector._update_tracks([bar], 10.0) == []  # tentative, not in state
    state = BattleState(10.0, GamePhase.BATTLE_1X, 10.0, [], units=[])
    assert runner.submit_action(action, state)
    assert runner._pending_deploy[3] is None


def test_identity_memory_is_separate_by_device_and_frame_size(tmp_path):
    def make(serial):
        cfg = BotConfig(identity_dir=str(tmp_path), adb_serial=serial, enable_identity_learning=True)
        return ClashRoyaleBotRunner(cfg, show_hud=False, auto_calibrate=False, manual_input=False)
    first, second = make("device-a"), make("device-b")
    assert first.identity.namespace != second.identity.namespace
    assert "device-a" not in first.identity.namespace
    first.identity.add_sample("knight", new_frame()[50:110, 50:110])
    first.identity.save()
    assert second.identity.sample_count == 0
    # Automatic device discovery and changed capture dimensions select a fresh namespace.
    first.adb.serial = "device-c"
    first.grid.resize(600, 1000)
    first._refresh_identity_memory()
    assert first.identity.sample_count == 0
    assert first.identity.namespace.startswith("600x1000_")
    assert first.arena_detector.cv_fallback.identity is first.identity


def test_deploy_verification_uses_local_grid_without_resizing_vision_grid(runner):
    class Capture:
        def capture_frame(self, timeout):
            return np.zeros((640, 360, 3), dtype=np.uint8)

    class Verifier:
        def snapshot(self, frame, grid):
            assert (grid.width, grid.height) == (360, 640)
            return "snapshot"

        def slot_changed(self, before, after, slot):
            return True

    runner.adb, runner.verifier = Capture(), Verifier()
    original_size = (runner.grid.width, runner.grid.height)
    before = runner._hand_snapshot()
    assert before == "snapshot"
    assert runner._wait_for_hand_change(0, before)
    assert (runner.grid.width, runner.grid.height) == original_size


def test_invalid_identity_samples_are_ignored_instead_of_returning_nan_match(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, knight__0=np.full((48, 40), np.nan, dtype=np.float32),
                        giant__0=np.zeros((3, 3), dtype=np.float32))
    memory = UnitIdentityMemory(tmp_path, "bad")
    assert memory.sample_count == 0
    assert memory.match(new_frame()[50:110, 50:110]) is None


def test_identity_save_is_atomic_when_replacement_fails(tmp_path, monkeypatch):
    import cr_bot.vision.unit_identity as identity_module

    memory = UnitIdentityMemory(tmp_path, "atomic")
    frame = new_frame()
    assert memory.add_sample("knight", frame[0:60, 0:60])
    memory.save()
    assert memory.add_sample("giant", frame[80:140, 80:140])
    monkeypatch.setattr(identity_module.os, "replace",
                        lambda *args: (_ for _ in ()).throw(OSError("simulated write failure")))
    with pytest.raises(OSError, match="simulated"):
        memory.save()
    reloaded = UnitIdentityMemory(tmp_path, "atomic")
    assert reloaded.sample_counts() == {"knight": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["atomic.npz"]
