"""Hybrid Arena Troop Detector: YOLOv8 / ONNX + OpenCV Health-Bar Fallback & Velocity Tracker."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaGrid
from cr_bot.config.cards_db import get_card_info, normalize_card_name
from cr_bot.models import DetectedUnit, MoveType, Side, WeightClass
from cr_bot.vision.unit_detector import (
    BlobUnit,
    BlobUnitDetector,
    DetectorStats,
    HealthBar,
    HealthBarReader,
    classify_size,
    pixels_per_tile,
)

if TYPE_CHECKING:  # pragma: no cover
    from cr_bot.vision.unit_identity import UnitIdentityMemory

logger = logging.getLogger(__name__)


@dataclass
class RawDetection:
    name: str
    side: Side
    tile_x: float
    tile_y: float
    pixel_bbox: Tuple[int, int, int, int]
    confidence: float
    hp_ratio: float
    move_type: MoveType
    weight_class: WeightClass
    source: str


class YOLOBackend:
    """Loads Ultralytics .pt or ONNX .onnx Clash Royale troop detector if weights are available."""

    def __init__(self, weights_path: Optional[str], conf_threshold: float = 0.42) -> None:
        self.conf_threshold = conf_threshold
        self.mode: str = "disabled"
        self._ultra_model = None
        self._ort_session = None
        self._cv_net = None
        self._class_names: Dict[int, str] = {}

        if not weights_path:
            return
        wpath = Path(weights_path)
        if not wpath.is_file():
            logger.info(
                "YOLO weights '%s' not found; using the built-in detector: background segmentation "
                "+ real-game sprite matching + velocity tracking (no ML weights needed).",
                wpath,
            )
            return

        suffix = wpath.suffix.lower()
        if suffix == ".pt":
            try:
                from ultralytics import YOLO  # type: ignore

                self._ultra_model = YOLO(str(wpath))
                self._class_names = dict(self._ultra_model.names)
                self.mode = "ultralytics"
                logger.info("Loaded Ultralytics YOLOv8 model: %s (%d classes)", wpath, len(self._class_names))
            except Exception as exc:
                logger.warning("Failed to load .pt weights (%s); falling back to OpenCV.", exc)
        elif suffix == ".onnx":
            self._load_sidecar_names(wpath)
            try:
                import onnxruntime as ort  # type: ignore

                self._ort_session = ort.InferenceSession(
                    str(wpath), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                self.mode = "onnxruntime"
                logger.info("Loaded ONNX Runtime model: %s", wpath)
            except Exception:
                try:
                    self._cv_net = cv2.dnn.readNetFromONNX(str(wpath))
                    self.mode = "opencv_dnn"
                    logger.info("Loaded ONNX model via OpenCV DNN: %s", wpath)
                except Exception as exc:
                    logger.warning("Failed to load ONNX weights (%s); falling back to OpenCV.", exc)

    def _load_sidecar_names(self, onnx_path: Path) -> None:
        names_file = onnx_path.with_suffix(".names")
        if names_file.is_file():
            lines = [ln.strip() for ln in names_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self._class_names = {i: ln for i, ln in enumerate(lines)}

    @property
    def is_active(self) -> bool:
        return self.mode != "disabled"

    def detect(self, frame_bgr: np.ndarray, grid: ArenaGrid) -> List[RawDetection]:
        if self.mode == "ultralytics" and self._ultra_model is not None:
            return self._detect_ultralytics(frame_bgr, grid)
        if self.mode in ("onnxruntime", "opencv_dnn"):
            return self._detect_onnx(frame_bgr, grid)
        return []

    def _detect_ultralytics(self, frame_bgr: np.ndarray, grid: ArenaGrid) -> List[RawDetection]:
        results = self._ultra_model.predict(frame_bgr, conf=self.conf_threshold, verbose=False)
        out: List[RawDetection] = []
        if not results:
            return out
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().item())
            cls_id = int(boxes.cls[i].cpu().item())
            raw_label = self._class_names.get(cls_id, f"class_{cls_id}")
            det = self._label_to_detection(raw_label, tuple(xyxy), conf, frame_bgr, grid)
            if det is not None:
                out.append(det)
        return out

    def _detect_onnx(self, frame_bgr: np.ndarray, grid: ArenaGrid) -> List[RawDetection]:
        h, w = frame_bgr.shape[:2]
        img_size = 640
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0 / 255.0, (img_size, img_size), swapRB=True, crop=False)

        if self.mode == "onnxruntime" and self._ort_session is not None:
            inp_name = self._ort_session.get_inputs()[0].name
            preds = self._ort_session.run(None, {inp_name: blob})[0]
        elif self._cv_net is not None:
            self._cv_net.setInput(blob)
            preds = self._cv_net.forward()
        else:
            return []

        # Standard YOLOv8 output shape: (1, 4 + num_classes, 8400) -> transpose to (8400, 4 + C)
        arr = np.squeeze(preds)
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T

        if arr.ndim != 2 or arr.shape[1] < 5:
            return []

        boxes_xywh = arr[:, :4]
        scores_all = arr[:, 4:]
        cls_ids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(len(arr)), cls_ids]

        keep = confs >= self.conf_threshold
        boxes_xywh = boxes_xywh[keep]
        cls_ids = cls_ids[keep]
        confs = confs[keep]

        if len(boxes_xywh) == 0:
            return []

        scale_x, scale_y = w / float(img_size), h / float(img_size)
        nms_boxes: List[List[int]] = []
        for cx, cy, bw, bh in boxes_xywh:
            x1 = int((cx - 0.5 * bw) * scale_x)
            y1 = int((cy - 0.5 * bh) * scale_y)
            nms_boxes.append([x1, y1, int(bw * scale_x), int(bh * scale_y)])

        indices = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), self.conf_threshold, 0.45)
        out: List[RawDetection] = []
        for idx in np.array(indices).flatten():
            bx, by, bw, bh = nms_boxes[int(idx)]
            bbox = (max(0, bx), max(0, by), min(w, bx + bw), min(h, by + bh))
            raw_label = self._class_names.get(int(cls_ids[int(idx)]), "unknown_medium")
            det = self._label_to_detection(raw_label, bbox, float(confs[int(idx)]), frame_bgr, grid)
            if det is not None:
                out.append(det)
        return out

    def _label_to_detection(
        self,
        raw_label: str,
        bbox: Tuple[int, int, int, int],
        conf: float,
        frame_bgr: np.ndarray,
        grid: ArenaGrid,
    ) -> Optional[RawDetection]:
        lower = raw_label.lower()
        side: Optional[Side] = None
        cleaned = lower
        for prefix, s_val in (
            ("enemy_", Side.ENEMY),
            ("opp_", Side.ENEMY),
            ("red_", Side.ENEMY),
            ("ally_", Side.ALLY),
            ("friendly_", Side.ALLY),
            ("blue_", Side.ALLY),
        ):
            if cleaned.startswith(prefix):
                side = s_val
                cleaned = cleaned[len(prefix) :]
                break
        if cleaned.endswith("_enemy") or cleaned.endswith("_1"):
            side = Side.ENEMY
            cleaned = cleaned.rsplit("_", 1)[0]
        elif cleaned.endswith("_ally") or cleaned.endswith("_0"):
            side = Side.ALLY
            cleaned = cleaned.rsplit("_", 1)[0]

        if side is None:
            side = OpenCVHealthBarBackend.infer_side_around_bbox(frame_bgr, bbox)

        card_slug = normalize_card_name(cleaned)
        info = get_card_info(card_slug)
        x1, y1, x2, y2 = bbox
        foot_px = 0.5 * (x1 + x2)
        foot_py = y2 - 0.15 * (y2 - y1)
        tx, ty = grid.pixel_to_tile(foot_px, foot_py)

        return RawDetection(
            name=info.name,
            side=side,
            tile_x=tx,
            tile_y=ty,
            pixel_bbox=bbox,
            confidence=conf,
            hp_ratio=1.0,
            move_type=info.move_type,
            weight_class=info.weight_class,
            source="yolo",
        )


class ArenaUnitBackend:
    """Segmentation-first troop detector for real devices (``cr_bot.vision.unit_detector``).

    The old implementation looked for red/blue horizontal strips and guessed the unit type from
    the strip's pixel width using 720p emulator constants.  This one segments moving objects out
    of the static arena, which yields a real bounding box, a tile-space position, a tile-relative
    size class, and a team side; health bars are then only used to refine side and HP.

    Identity is optional: :class:`cr_bot.vision.unit_identity.UnitIdentityMemory` supplies card
    names once the bot has seen enough of its own verified deploys.  Without it, detections keep
    the size-based ``unknown_*`` label, which is exactly what the brain needs to pick a counter.
    """

    #: how far below a health bar the unit stands, in tiles (used before segmentation warms up)
    BAR_TO_FOOT_TILES = 0.82

    def __init__(
        self,
        identity_memory: Optional["UnitIdentityMemory"] = None,
        enable_segmentation: bool = True,
        enable_identity: bool = True,
    ) -> None:
        self.segmenter = BlobUnitDetector()
        self.bar_reader = HealthBarReader()
        self.identity = identity_memory
        self.enable_segmentation = enable_segmentation
        self.enable_identity = enable_identity
        self.last_bars: List[HealthBar] = []
        self.stats = DetectorStats()
        #: how many appearance samples have been learned from verified (own) deployments
        self.learned_samples: int = 0
        #: consecutive frames in which only bars were found (see RESET_AFTER_STALE_FRAMES)
        self._stale_frames: int = 0

    def reset(self) -> None:
        self.segmenter.reset()
        self.stats = DetectorStats()
        self._stale_frames = 0

    # -- main entry ----------------------------------------------------------------
    #: a segmentation blob this uncertain, with no health bar to confirm it, is dropped
    MIN_CONFIDENCE = 0.30
    #: two detections of the same side this close together are one unit, not two
    DEDUPE_TILES = 0.75

    #: consecutive frames in which only bars (never silhouettes) were found before the background
    #: model is told to re-learn the arena
    RESET_AFTER_STALE_FRAMES = 4

    #: how much confidence a bars-only detection loses. It is the weakest evidence class: no
    #: silhouette, no size measurement, no side vote — just "a bar-shaped capsule is here".
    BARS_ONLY_CONFIDENCE = 0.45

    def detect(self, frame_bgr: np.ndarray, grid: ArenaGrid) -> List[RawDetection]:
        units: List[BlobUnit] = []
        if self.enable_segmentation:
            units = self.segmenter.detect(frame_bgr, grid)
            self.last_bars = self.segmenter.last_bars
        if not units:
            # Nothing moving was found: either the background model is still warming up or the
            # arena is genuinely empty. Bars cover both cases (and are the only signal that
            # survives a stationary unit on the very first frames).
            return self._detect_bars_only(frame_bgr, grid)

        described = [u for u in units if self._keep(u)]
        out = [self._describe(unit, frame_bgr, grid) for unit in described]
        # A troop that the background model has not learned yet — it is standing still, or it
        # just walked into a part of the arena the model has not settled — still draws its bar.
        # Adding those as low-confidence detections makes a unit appear immediately instead of
        # seconds later, but they must always lose to a real silhouette, so anything segmentation
        # found within a tile of them wins.
        extras = self._detect_bars_only(frame_bgr, grid)
        for extra in extras:
            if any(
                math.hypot(extra.tile_x - d.tile_x, extra.tile_y - d.tile_y) <= 1.0
                and extra.side == d.side
                for d in out
            ):
                continue
            out.append(extra)

        # Nothing from segmentation while a unit's bar *is* on the board means the background
        # model has gone stale — typically because the arena changed under it (a tower fell, the
        # camera shifted a pixel, a unit parked on scenery and was absorbed). Telling it to
        # re-learn brings the silhouettes back within a few frames. The gate matters: an empty
        # arena also produces no segmentation, and resetting on that would keep the model in
        # warm-up forever and detect nothing at all.
        if out and len(out) == len(extras):
            self._stale_frames += 1
            if self._stale_frames >= self.RESET_AFTER_STALE_FRAMES and self.enable_segmentation:
                logger.info(
                    "detector: %d frame(s) with bars but no silhouettes; re-learning the arena",
                    self._stale_frames,
                )
                self.segmenter.background.reset()
                self._stale_frames = 0
        else:
            self._stale_frames = 0

        out = self._dedupe(out, grid)
        self._remember(out, frame_bgr, grid)
        self.stats.add(described, bars=len(self.last_bars),
                       identified=sum(1 for d in out if not d.name.startswith("unknown_")))
        return out

    def _keep(self, unit: BlobUnit) -> bool:
        """Drop detections that are not backed by enough evidence to act on.

        A blob can be a troop, or it can be a spell splat, a spawn animation, a Bush that just
        rustled, or the dust from a destroyed tower. The old code accepted all of them, which
        is where "detected as multiple troops" and "phantom enemies" came from. A health bar is
        proof of a real unit, so a bar-backed blob is always kept.
        """
        if unit.has_bar:
            return True
        return unit.confidence >= self.MIN_CONFIDENCE

    def _dedupe(self, detections: List[RawDetection], grid: ArenaGrid) -> List[RawDetection]:
        """Collapse two detections of the same side that describe one unit.

        Segmentation and the bar fallback can both find the same troop, and a sprite can flicker
        into two blobs when a limb detaches for one frame. Keeping both makes the board look
        twice as threatening as it is, which is exactly how a bot wastes elixir.
        """
        ordered = sorted(
            detections,
            key=lambda d: (d.hp_ratio != 1.0, d.confidence, d.weight_class != WeightClass.SWARM),
            reverse=True,
        )
        kept: List[RawDetection] = []
        for det in ordered:
            duplicate = False
            for other in kept:
                if other.side != det.side:
                    continue
                dist = math.hypot(other.tile_x - det.tile_x, other.tile_y - det.tile_y)
                if dist <= self.DEDUPE_TILES and (
                    other.name == det.name or det.name.startswith("unknown_")
                    or other.name.startswith("unknown_")
                ):
                    duplicate = True
                    # let the confirmed one win, but keep the best confidence
                    if det.confidence > other.confidence:
                        other.confidence = det.confidence
                    break
            if not duplicate:
                kept.append(det)
        return kept

    def _describe(self, unit: BlobUnit, frame_bgr: np.ndarray, grid: ArenaGrid) -> RawDetection:
        name = unit.fallback_label
        move_type = MoveType.GROUND
        weight = unit.weight_class
        # confidence now reflects how much *evidence* there is (segmentation solidity, team
        # vote, bar support) instead of a fixed 0.55-0.8 that made every guess look solid.
        confidence = float(unit.confidence)
        source = "segmentation"

        if self.enable_identity and self.identity is not None and self.identity.sample_count:
            patch = self._patch(frame_bgr, unit.bbox)
            match = self.identity.match(patch)
            if match is not None:
                name = match.card
                source = "segmentation+identity"
                confidence = float(min(0.97, 0.55 + 0.45 * max(0.0, match.score)))
                info = get_card_info(match.card)
                if info is not None:
                    move_type = info.move_type
                    if info.weight_class not in (WeightClass.SPELL_PROJECTILE, WeightClass.BUILDING):
                        weight = info.weight_class

        return RawDetection(
            name=name,
            side=unit.side,
            tile_x=unit.tile_x,
            tile_y=unit.tile_y,
            pixel_bbox=unit.bbox,
            confidence=round(confidence, 2),
            hp_ratio=round(float(unit.hp_ratio), 2),
            move_type=move_type,
            weight_class=weight,
            source=source,
        )

    @staticmethod
    def _patch(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame_bgr.shape[1], x2)
        y2 = min(frame_bgr.shape[0], y2)
        return frame_bgr[y1:y2, x1:x2]

    #: a sample is only stored when the detection is this well supported
    LEARN_MIN_CONFIDENCE = 0.80
    #: only learn from our own units: we know what we deployed, so the label is verifiable
    LEARN_SIDES = ("ally",)

    def _remember(self, detections: List[RawDetection], frame_bgr: np.ndarray,
                  grid: ArenaGrid) -> None:
        """Learn a card's appearance — but only from detections whose *label is verified*.

        This used to store every confident detection, which turned a single mistake into a
        permanent one: a Wizard read as a Musketeer was written into the library as a Musketeer
        sample, and from then on the matcher agreed with the mistake. Worse, a detection that
        already *came from* the library was fed straight back in, so a wrong label could
        reinforce itself frame after frame.

        The label is only trustworthy when we produced it ourselves: an **ally** unit is one the
        bot just deployed, so its card is known from the hand, not guessed. Enemy detections are
        never used as training data. On top of that the detection must be clean — full HP (no
        damage tint or overlapping health bar) and strongly supported by the detector (a bar or
        good segmentation).
        """
        if self.identity is None or not self.enable_identity:
            return
        for det in detections:
            if det.name.startswith("unknown_"):
                continue
            if getattr(det.side, "value", str(det.side)) not in self.LEARN_SIDES:
                continue
            if det.confidence < self.LEARN_MIN_CONFIDENCE:
                continue
            if det.hp_ratio < 0.99:
                continue      # damaged sprite: bar and damage flash pollute the patch
            if det.source not in ("segmentation", "segmentation+identity"):
                continue
            if self.identity.add_from_frame(det.name, frame_bgr, det.pixel_bbox):
                self.learned_samples += 1

    # -- fallback while the background model warms up ------------------------------
    #: A fallback bar must be at least this wide (tiles). Measured on the real frame: every
    #: troop's bar is >= 1.1 tiles even when white HP digits cut it in half, while the pieces of
    #: UI text and decoration that also survive the shape filters are all under 0.5 tiles.
    #: The fallback's job is to keep *substantial* units visible while segmentation warms up;
    #: a 0.3-tile strip is not reliable evidence of anything, so it is not reported at all.
    MIN_FALLBACK_BAR_TILES = 0.90

    def _detect_bars_only(self, frame_bgr: np.ndarray, grid: ArenaGrid) -> List[RawDetection]:
        """Last-resort detection from health bars alone.

        This runs when segmentation has nothing to say: the first frames of a match, or a frame
        where everything is standing still. It reads the *bars* the game draws above units.

        The trap here is that tower HP bars and tower decorations look exactly like unit bars,
        and a permissive reader finds two dozen of them on a real frame. Reporting those as
        troops is the "detected as multiple troops" complaint in its purest form: the brain sees
        an army that is not there and empties its elixir. So the fallback is strict — wide bars
        only, towers excluded — and its output is marked low confidence.
        """
        ppt = pixels_per_tile(grid)
        bars = self.bar_reader.read(frame_bgr, grid, ppt,
                                    min_width_tiles=self.MIN_FALLBACK_BAR_TILES)
        self.last_bars = bars

        out: List[RawDetection] = []
        foot_offset = self.BAR_TO_FOOT_TILES * ppt
        # Where the arena actually ends. The bar reader searches the whole calibrated arena
        # rectangle, which on a real phone extends over the card tray and the elixir bar; the
        # *play field* is the part above the river-and-lane strip the game reserves for the HUD.
        # Reading bars there reported the elixir pips and the hand's level badges as troops: on
        # one real frame, four of seven "units" sat at y ~ 31 tiles, i.e. in the tray.
        _ax1, _ay1, _ax2, field_bottom = self._field_bounds(frame_bgr, grid)
        out_of_field = 0
        for bar in bars:
            x1, _y1, x2, y2 = bar.bbox
            if y2 + foot_offset > field_bottom:
                out_of_field += 1
                continue
            # Only a bar with the game's dark UI capsule around it is a unit's HP bar; a bare
            # coloured strip is a tower ring or a piece of decoration.
            if not getattr(bar, "has_outline", True):
                continue
            cx = 0.5 * (x1 + x2)
            foot_y = y2 + foot_offset
            tx, ty = grid.pixel_to_tile(cx, foot_y)
            if not (0.0 <= tx <= grid.TILES_X and 0.0 <= ty <= grid.TILES_Y):
                continue
            if self._near_tower(grid, tx, ty):
                continue    # tower HP bar, not a troop
            if not self._inside_playfield(grid, ty):
                # Above the enemy king and below your own king there is no playfield: that is
                # the enemy's castle wall and name plate at the top, and the wall plus the card
                # tray at the bottom. Bars there belong to neither team's troops.
                continue
            width_tiles = bar.total_width / ppt
            weight, label = classify_size(max(0.5, width_tiles * 1.6))
            out.append(
                RawDetection(
                    name=label,
                    side=bar.side,
                    tile_x=float(tx),
                    tile_y=float(ty),
                    pixel_bbox=(int(x1 - 0.5 * ppt), int(y2), int(x2 + 0.5 * ppt), int(foot_y)),
                    confidence=0.45,
                    hp_ratio=round(float(bar.hp_ratio), 2),
                    move_type=MoveType.GROUND,
                    weight_class=weight,
                    source="bars",
                )
            )
        if out_of_field:
            logger.debug("bars-only fallback: ignored %d bar(s) below the play field", out_of_field)
        return self._dedupe(out, grid)

    #: the fraction of the calibrated arena rectangle that is actually play field (the rest is
    #: the card tray / elixir strip at the bottom and the enemy's name plate at the top)
    FIELD_BOTTOM_FRACTION = 0.90
    FIELD_TOP_FRACTION = 0.03

    def _field_bounds(self, frame_bgr: np.ndarray, grid: ArenaGrid):
        """Pixel rectangle of the *playable* field inside the calibrated arena area."""
        ax1, ay1, ax2, ay2 = grid.arena_pixel_bounds
        height = float(ay2 - ay1)
        top = int(ay1 + self.FIELD_TOP_FRACTION * height)
        bottom = int(ay1 + self.FIELD_BOTTOM_FRACTION * height)
        h, w = frame_bgr.shape[:2]
        return (max(0, ax1), max(0, top), min(w, ax2), min(h, bottom))

    #: margin (tiles) inside the kings' rows that still counts as playfield
    PLAYFIELD_MARGIN_TILES = 1.0

    @classmethod
    def _inside_playfield(cls, grid: ArenaGrid, tile_y: float) -> bool:
        """Is this row of the arena actually playable? (the kings' rows bracket the playfield)"""
        towers = getattr(grid, "TOWER_TILES", {})
        enemy_king = towers.get("enemy_king")
        ally_king = towers.get("ally_king")
        if not towers:
            return True
        if enemy_king is not None and tile_y < enemy_king[1] - cls.PLAYFIELD_MARGIN_TILES:
            return False
        if ally_king is not None and tile_y > ally_king[1] - cls.PLAYFIELD_MARGIN_TILES:
            return False
        return True

    #: a tower's own HP bar is drawn on its crown, just above and around the tower centre
    TOWER_BAR_RADIUS_TILES = 1.2
    #: ... and a troop fighting *next to* a tower is 1-2 tiles away, so the vertical test has to
    #: be one-sided: only a bar at or above the crown can be the tower's own
    TOWER_BAR_MAX_BELOW_TILES = 0.6

    @classmethod
    def _near_tower(cls, grid: ArenaGrid, tile_x: float, tile_y: float,
                    radius_tiles: Optional[float] = None) -> bool:
        """True when this *bar* belongs to a tower rather than to a troop.

        The old test was a plain 2.2-tile circle, which is wider than the gap between a tower
        and the troop attacking it — so an enemy Mega Knight fighting at the princess tower was
        silently discarded as "the tower's bar". The radius is now the tower's own footprint and
        the bar must be drawn at or above the crown, which is where the game puts it.
        """
        radius = cls.TOWER_BAR_RADIUS_TILES if radius_tiles is None else radius_tiles
        for tower_x, tower_y in grid.TOWER_TILES.values():
            if math.hypot(tile_x - tower_x, tile_y - tower_y) > radius:
                continue
            if tile_y <= tower_y + cls.TOWER_BAR_MAX_BELOW_TILES:
                return True
        return False

    @staticmethod
    def infer_side_around_bbox(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Side:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, _ = bbox
        head = frame_bgr[max(0, y1 - 18): min(h, y1 + 14), max(0, x1 - 6): min(w, x2 + 6)]
        if head.size == 0:
            return Side.ENEMY
        hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array([0, 110, 110]), np.array([10, 255, 255])) | cv2.inRange(
            hsv, np.array([168, 110, 110]), np.array([179, 255, 255])
        )
        blue = cv2.inRange(hsv, np.array([92, 100, 110]), np.array([118, 255, 255]))
        return Side.ALLY if np.count_nonzero(blue) > np.count_nonzero(red) else Side.ENEMY


#: kept for backwards compatibility with older scripts/configs
OpenCVHealthBarBackend = ArenaUnitBackend


class HybridArenaDetector:
    """Fuses YOLOv8/ONNX detections with OpenCV health-bar detections and tracks unit velocities."""

    MAX_MATCH_TILE_DIST = 2.7
    #: How long a track survives without being re-detected.
    #:
    #: This used to be 0.35s, and every blink of the detector — one dropped frame, a unit
    #: passing under a tower, a spell flash — killed the track. The next frame re-detected the
    #: same troop with a *new* track id, which resets ``first_seen_ts`` (so time-based logic
    #: like the Inferno reset window restarts), resets the smoothed velocity to zero, and makes
    #: the brain's elixir/cycle accounting count the same enemy card again. A track must
    #: outlive a few missed frames instead.
    TRACK_TTL_SEC = 1.6
    #: a track that has not been seen for this long is reported with reduced confidence
    STALE_AFTER_SEC = 0.45

    def __init__(
        self,
        grid: ArenaGrid,
        yolo_weights_path: Optional[str] = None,
        yolo_conf: float = 0.42,
        enable_opencv_fallback: bool = True,
        identity_memory: Optional["UnitIdentityMemory"] = None,
        enable_segmentation: bool = True,
    ) -> None:
        self.grid = grid
        self.yolo = YOLOBackend(yolo_weights_path, conf_threshold=yolo_conf)
        self.cv_fallback = (
            ArenaUnitBackend(identity_memory=identity_memory, enable_segmentation=enable_segmentation)
            if enable_opencv_fallback
            else None
        )
        self._tracks: Dict[int, DetectedUnit] = {}
        self._next_track_id: int = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        if self.cv_fallback is not None:
            self.cv_fallback.reset()

    def detect_and_track(self, frame_bgr: np.ndarray, now_ts: float) -> List[DetectedUnit]:
        yolo_dets = self.yolo.detect(frame_bgr, self.grid) if self.yolo.is_active else []
        cv_dets = self.cv_fallback.detect(frame_bgr, self.grid) if self.cv_fallback else []

        fused = self._fuse_detections(yolo_dets, cv_dets)
        return self._update_tracks(fused, now_ts)

    @staticmethod
    def _fuse_detections(yolo_dets: List[RawDetection], cv_dets: List[RawDetection]) -> List[RawDetection]:
        if not yolo_dets:
            return cv_dets
        if not cv_dets:
            return yolo_dets

        merged: List[RawDetection] = []
        used_cv = set()
        for yd in yolo_dets:
            best_cv_idx = None
            best_dist = 1.6
            for i, cd in enumerate(cv_dets):
                if i in used_cv or cd.side != yd.side:
                    continue
                d = math.hypot(yd.tile_x - cd.tile_x, yd.tile_y - cd.tile_y)
                if d < best_dist:
                    best_dist = d
                    best_cv_idx = i
            if best_cv_idx is not None:
                used_cv.add(best_cv_idx)
                cd = cv_dets[best_cv_idx]
                merged.append(
                    RawDetection(
                        name=yd.name,
                        side=yd.side,
                        tile_x=yd.tile_x,
                        tile_y=yd.tile_y,
                        pixel_bbox=yd.pixel_bbox,
                        confidence=yd.confidence,
                        hp_ratio=cd.hp_ratio,
                        move_type=yd.move_type,
                        weight_class=yd.weight_class,
                        source="fused",
                    )
                )
            else:
                merged.append(yd)

        # Keep any unmatched OpenCV health-bar detections so zero enemy pushes slip through unseen!
        for i, cd in enumerate(cv_dets):
            if i not in used_cv:
                merged.append(cd)
        return merged

    def _update_tracks(self, detections: List[RawDetection], now_ts: float) -> List[DetectedUnit]:
        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = set(range(len(detections)))

        # Greedy nearest-neighbor matching in tile space within same Side
        pairs: List[Tuple[float, int, int]] = []
        for tid in unmatched_tracks:
            tr = self._tracks[tid]
            for didx in unmatched_dets:
                det = detections[didx]
                if tr.side != det.side:
                    continue
                dist = math.hypot(tr.tile_x - det.tile_x, tr.tile_y - det.tile_y)
                if dist <= self.MAX_MATCH_TILE_DIST:
                    pairs.append((dist, tid, didx))

        pairs.sort(key=lambda x: x[0])
        for _, tid, didx in pairs:
            if tid not in unmatched_tracks or didx not in unmatched_dets:
                continue
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(didx)

            tr = self._tracks[tid]
            det = detections[didx]
            dt = max(0.04, now_ts - tr.last_seen_ts)
            inst_vx = (det.tile_x - tr.tile_x) / dt
            inst_vy = (det.tile_y - tr.tile_y) / dt

            # Exponential moving average velocity smoothing
            tr.vx = round(0.55 * tr.vx + 0.45 * inst_vx, 2)
            tr.vy = round(0.55 * tr.vy + 0.45 * inst_vy, 2)
            tr.tile_x = round(det.tile_x, 2)
            tr.tile_y = round(det.tile_y, 2)
            tr.pixel_bbox = det.pixel_bbox
            tr.hp_ratio = det.hp_ratio
            tr.confidence = det.confidence
            tr.last_seen_ts = now_ts
            tr.source = det.source

            # Preserve specific YOLO identity if fallback only returned generic `unknown_*`
            if not det.name.startswith("unknown_"):
                tr.name = det.name
                tr.move_type = det.move_type
                tr.weight_class = det.weight_class
            elif tr.name.startswith("unknown_"):
                # Upgrade generic classification using observed speed/river crossing!
                self._refine_generic_class(tr)

        # Spawn new tracks for unmatched detections
        for didx in unmatched_dets:
            det = detections[didx]
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = DetectedUnit(
                track_id=tid,
                name=det.name,
                side=det.side,
                tile_x=round(det.tile_x, 2),
                tile_y=round(det.tile_y, 2),
                pixel_bbox=det.pixel_bbox,
                confidence=det.confidence,
                hp_ratio=det.hp_ratio,
                move_type=det.move_type,
                weight_class=det.weight_class,
                vx=0.0,
                vy=0.0,
                first_seen_ts=now_ts,
                last_seen_ts=now_ts,
                source=det.source,
            )

        # Expire stale tracks
        expired = [
            tid for tid, tr in self._tracks.items() if (now_ts - tr.last_seen_ts) > self.TRACK_TTL_SEC
        ]
        for tid in expired:
            del self._tracks[tid]

        # Report live tracks; ones that were missed for a few frames stay in the board (so the
        # brain does not forget a push mid-fight) but are marked less certain.
        out: List[DetectedUnit] = []
        for tr in self._tracks.values():
            missed = now_ts - tr.last_seen_ts
            if missed > self.TRACK_TTL_SEC:
                continue
            if missed > self.STALE_AFTER_SEC:
                tr.confidence = round(max(0.2, tr.confidence * 0.92), 3)
                # a stale track cannot be trusted to still be moving
                tr.vx *= 0.6
                tr.vy *= 0.6
            out.append(tr)
        return out

    @staticmethod
    def _refine_generic_class(unit: DetectedUnit) -> None:
        """Infer air vs ground rusher on OpenCV-only tracks from river position & speed."""
        speed = math.hypot(unit.vx, unit.vy)
        # Crossing river far from bridges (bridges are at x~3.5 and x~14.5) => must be AIR or Miner/Barrel!
        in_river_band = 14.8 <= unit.tile_y <= 17.2
        away_from_bridges = not (2.0 <= unit.tile_x <= 5.2 or 12.8 <= unit.tile_x <= 16.0)
        if in_river_band and away_from_bridges and speed >= 1.2:
            unit.move_type = MoveType.AIR
            unit.name = "unknown_air"
