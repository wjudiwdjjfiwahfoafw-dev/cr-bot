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

    def reset(self) -> None:
        self.segmenter.reset()
        self.stats = DetectorStats()

    # -- main entry ----------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray, grid: ArenaGrid) -> List[RawDetection]:
        units: List[BlobUnit] = []
        if self.enable_segmentation:
            units = self.segmenter.detect(frame_bgr, grid)
            self.last_bars = self.segmenter.last_bars
        if not units:
            # Nothing moving was found: either the background model is still warming up or the
            # arena is genuinely empty (a spell lands, an elixir bar animates). Bars cover both.
            return self._detect_bars_only(frame_bgr, grid)
        out = [self._describe(unit, frame_bgr, grid) for unit in units]
        self._remember(out, frame_bgr, grid)
        self.stats.add(units, bars=len(self.last_bars),
                       identified=sum(1 for d in out if not d.name.startswith("unknown_")))
        return out

    def _describe(self, unit: BlobUnit, frame_bgr: np.ndarray, grid: ArenaGrid) -> RawDetection:
        name = unit.fallback_label
        move_type = MoveType.GROUND
        weight = unit.weight_class
        confidence = 0.55 + 0.25 * min(1.0, unit.side_confidence)
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

    def _remember(self, detections: List[RawDetection], frame_bgr: np.ndarray, grid: ArenaGrid) -> None:
        """Learn confirmed identities: a named detection strengthens its own reference set."""
        if self.identity is None or not self.enable_identity:
            return
        for det in detections:
            if det.name.startswith("unknown_") or det.confidence < 0.7:
                continue
            self.identity.add_from_frame(det.name, frame_bgr, det.pixel_bbox)

    # -- fallback while the background model warms up ------------------------------
    def _detect_bars_only(self, frame_bgr: np.ndarray, grid: ArenaGrid) -> List[RawDetection]:
        ppt = pixels_per_tile(grid)
        bars = self.segmenter.last_bars or self.bar_reader.read(frame_bgr, grid, ppt)
        self.last_bars = bars
        out: List[RawDetection] = []
        foot_offset = self.BAR_TO_FOOT_TILES * ppt
        for bar in bars:
            x1, _y1, x2, y2 = bar.bbox
            cx = 0.5 * (x1 + x2)
            foot_y = y2 + foot_offset
            tx, ty = grid.pixel_to_tile(cx, foot_y)
            if not (0.0 <= tx <= grid.TILES_X and 0.0 <= ty <= grid.TILES_Y):
                continue
            # a real device draws unit bars between ~0.2 and ~2.8 tiles wide
            width_tiles = bar.total_width / ppt
            weight, label = classify_size(max(0.5, width_tiles * 1.6))
            out.append(
                RawDetection(
                    name=label,
                    side=bar.side,
                    tile_x=float(tx),
                    tile_y=float(ty),
                    pixel_bbox=(int(x1 - 0.5 * ppt), int(y2), int(x2 + 0.5 * ppt), int(foot_y)),
                    confidence=0.5,
                    hp_ratio=round(float(bar.hp_ratio), 2),
                    move_type=MoveType.GROUND,
                    weight_class=weight,
                    source="bars",
                )
            )
        return out

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
    TRACK_TTL_SEC = 0.85

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

        # Return currently active or very recently seen (< 0.35s) units
        return [
            tr for tr in self._tracks.values() if (now_ts - tr.last_seen_ts) <= 0.35
        ]

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
