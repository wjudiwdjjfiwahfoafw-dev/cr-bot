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

    # Tactical tuning knobs for MK Zap Bait
    mk_min_stomp_elixir_value: float = 4.0
    leak_prevention_elixir: float = 9.3
    bait_window_seconds: float = 11.0
    overtime_spell_finish_hp: int = 380

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
        mk_min_stomp_elixir_value=float(tactics_cfg.get("mk_min_stomp_elixir_value", 4.0)),
        leak_prevention_elixir=float(tactics_cfg.get("leak_prevention_elixir", 9.3)),
        bait_window_seconds=float(tactics_cfg.get("bait_window_seconds", 11.0)),
        overtime_spell_finish_hp=int(tactics_cfg.get("overtime_spell_finish_hp", 380)),
    )
