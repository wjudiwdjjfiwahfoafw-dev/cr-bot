"""Configuration loader and schema defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from cr_bot.config.cards_db import get_card_info, normalize_card_name


@dataclass
class BotConfig:
    deck: List[str] = field(
        default_factory=lambda: [
            "mega_knight",
            "miner",
            "inferno_dragon",
            "skeleton_barrel",
            "zap",
            "bats",
            "spear_goblins",
            "goblin_gang",
        ]
    )
    evolutions: List[str] = field(default_factory=lambda: ["mega_knight", "bats"])
    adb_path: str = "adb"
    adb_serial: Optional[str] = None
    target_width: int = 720
    target_height: int = 1280
    loop_hz: float = 12.0
    min_action_cooldown_sec: float = 0.65
    human_jitter_px: int = 4
    auto_queue_battle: bool = False
    templates_dir: str = "assets/card_templates"
    yolo_weights_path: Optional[str] = "weights/clash_yolov8n.onnx"
    yolo_confidence: float = 0.42
    enable_opencv_fallback: bool = True
    tower_level: int = 11

    # --- real-device capture / input ------------------------------------------------
    capture_width: int = 720          # streamed frame width (lower = faster, still accurate)
    capture_backend: str = "auto"     # auto | screenrecord | scrcpy_py | screencap
    capture_bitrate: int = 12_000_000
    capture_max_fps: int = 30
    input_backend: str = "auto"       # auto | swipe | motionevent | tap
    input_backend_chain: List[str] = field(
        default_factory=lambda: ["swipe", "motionevent", "tap"]
    )
    deploy_style: str = "drag"        # drag | tap
    deploy_verify_threshold: float = 9.0
    calibration_dir: str = "configs/calibration"
    #: application ids that count as "the game" - custom builds (Nulls Royale, private servers)
    #: are the same game with a different id, so this is a list and never hard-coded
    game_packages: List[str] = field(
        default_factory=lambda: ["com.supercell.clashroyale", "nullsroyale.rel.free"]
    )
    game_package_hints: List[str] = field(
        default_factory=lambda: ["clashroyale", "clash_royale", "nullsroyale"]
    )
    #: explicit scrcpy server jar (default: newest bundled)
    identity_dir: str = "assets/unit_identity"
    enable_identity_learning: bool = True
    keep_screen_on: bool = True
    lock_portrait: bool = True
    adb_latency_sec: float = 0.10

    # Tactical tuning knobs for MK Zap Bait
    mk_min_stomp_elixir_value: float = 4.0
    leak_prevention_elixir: float = 9.3
    bait_window_seconds: float = 11.0
    overtime_spell_finish_hp: int = 380

    def input_backend_order(self) -> List[str]:
        """Backend preference chain, with the configured primary moved to the front."""
        chain = [b for b in self.input_backend_chain if b]
        primary = (self.input_backend or "auto").lower()
        if primary not in ("auto", "default", ""):
            chain = [primary] + [b for b in chain if b != primary]
        return chain or ["swipe", "motionevent", "tap"]

    def normalized_deck(self) -> List[str]:
        return [normalize_card_name(c) for c in self.deck]

    def average_elixir(self) -> float:
        cards = self.normalized_deck()
        if not cards:
            return 0.0
        return sum(get_card_info(c).elixir for c in cards) / len(cards)


def load_config(path: str | Path) -> BotConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return BotConfig()
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    deck_raw = raw.get("deck", BotConfig().deck)
    deck = [normalize_card_name(str(c)) for c in deck_raw]
    evos = [normalize_card_name(str(c)) for c in raw.get("evolutions", ["mega_knight", "bats"])]

    adb_cfg = raw.get("adb", {})
    vision_cfg = raw.get("vision", {})
    tactics_cfg = raw.get("tactics", {})
    capture_cfg = raw.get("capture", {})
    input_cfg = raw.get("input", {})
    calib_cfg = raw.get("calibration", {}) or {}
    game_cfg = raw.get("game", {}) or {}

    return BotConfig(
        deck=deck,
        evolutions=evos,
        adb_path=str(adb_cfg.get("adb_path", "adb")),
        adb_serial=adb_cfg.get("serial"),
        target_width=int(adb_cfg.get("target_width", 720)),
        target_height=int(adb_cfg.get("target_height", 1280)),
        loop_hz=float(raw.get("loop_hz", 12.0)),
        min_action_cooldown_sec=float(tactics_cfg.get("min_action_cooldown_sec", 0.65)),
        human_jitter_px=int(adb_cfg.get("human_jitter_px", 4)),
        auto_queue_battle=bool(raw.get("auto_queue_battle", False)),
        templates_dir=str(vision_cfg.get("templates_dir", "assets/card_templates")),
        yolo_weights_path=vision_cfg.get("yolo_weights_path", "weights/clash_yolov8n.onnx"),
        yolo_confidence=float(vision_cfg.get("yolo_confidence", 0.42)),
        enable_opencv_fallback=bool(vision_cfg.get("enable_opencv_fallback", True)),
        tower_level=int(tactics_cfg.get("tower_level", 11)),
        capture_width=int(capture_cfg.get("width", 720)),
        capture_backend=str(capture_cfg.get("backend", "auto")),
        capture_bitrate=int(capture_cfg.get("bitrate", 12_000_000)),
        capture_max_fps=int(capture_cfg.get("max_fps", 30)),
        input_backend=str(input_cfg.get("backend", "auto")),
        input_backend_chain=list(input_cfg.get("chain", ["swipe", "motionevent", "tap"])),
        deploy_style=str(input_cfg.get("deploy_style", "drag")),
        deploy_verify_threshold=float(input_cfg.get("verify_threshold", 9.0)),
        calibration_dir=str(calib_cfg.get("dir", "configs/calibration")),
        game_packages=list(game_cfg.get("packages", None) or
                           ["com.supercell.clashroyale", "nullsroyale.rel.free"]),
        game_package_hints=list(game_cfg.get("package_hints", None) or
                                ["clashroyale", "clash_royale", "nullsroyale"]),
        identity_dir=str(vision_cfg.get("identity_dir", "assets/unit_identity")),
        enable_identity_learning=bool(vision_cfg.get("identity_learning", True)),
        keep_screen_on=bool(adb_cfg.get("keep_screen_on", True)),
        lock_portrait=bool(adb_cfg.get("lock_portrait", True)),
        adb_latency_sec=float(adb_cfg.get("latency_sec", 0.10)),
        mk_min_stomp_elixir_value=float(tactics_cfg.get("mk_min_stomp_elixir_value", 4.0)),
        leak_prevention_elixir=float(tactics_cfg.get("leak_prevention_elixir", 9.3)),
        bait_window_seconds=float(tactics_cfg.get("bait_window_seconds", 11.0)),
        overtime_spell_finish_hp=int(tactics_cfg.get("overtime_spell_finish_hp", 380)),
    )
