"""
Mobile Phone Detector (with Remote Suppression & High Sensitivity)
==================================================================

- Detects COCO class 67 (cell phone) with instant 1-frame trigger
- Explicitly recognizes and suppresses COCO class 65 (remote)
- Provides lost-frame persistence to eliminate audio stutter/flicker
"""

import time
import sys
import types
import threading
import cv2
import numpy as np
import config


def _ensure_torch_nms_patch() -> None:
    """Fix 'operator torchvision::nms does not exist' across Python versions/Windows."""
    try:
        import torch

        def _pure_torch_nms(boxes, scores, iou_threshold):
            if boxes.numel() == 0:
                return torch.empty((0,), dtype=torch.long, device=boxes.device)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort(descending=True)
            keep = []
            while order.numel() > 0:
                if order.numel() == 1:
                    keep.append(order.item())
                    break
                i = order[0].item()
                keep.append(i)
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])
                w = torch.clamp(xx2 - xx1, min=0.0)
                h = torch.clamp(yy2 - yy1, min=0.0)
                inter = w * h
                union = areas[i] + areas[order[1:]] - inter
                iou = inter / torch.clamp(union, min=1e-6)
                order = order[1:][iou <= iou_threshold]
            return torch.tensor(keep, dtype=torch.long, device=boxes.device)

        try:
            import torchvision
            torchvision.ops.nms = _pure_torch_nms
        except Exception:
            pass

        if "torchvision.ops" in sys.modules:
            sys.modules["torchvision.ops"].nms = _pure_torch_nms
        else:
            tv = sys.modules.get("torchvision", types.ModuleType("torchvision"))
            tv_ops = types.ModuleType("torchvision.ops")
            tv_ops.nms = _pure_torch_nms
            tv.ops = tv_ops
            sys.modules["torchvision"] = tv
            sys.modules["torchvision.ops"] = tv_ops
    except Exception:
        pass


class PhoneDetector:
    """
    YOLOv8-based mobile phone detector with threaded background inference.
    Maintains 30+ FPS video stream while detecting phones instantly.
    """

    def __init__(
        self,
        model_path: str = str(config.YOLO_MODEL_PATH),
        conf_threshold: float = config.PHONE_DETECTION_CONFIDENCE,
        lost_frames: int = config.PHONE_LOST_FRAMES,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.lost_frames = lost_frames
        self.model_path = model_path
        self._model = None

        # Temporal State Tracking
        self._consecutive_detections = 0
        self._consecutive_misses = 0
        self._is_phone_confirmed = False
        self._last_valid_detections: list[dict] = []
        self._suppressed_objects: list[dict] = []
        self._lock = threading.Lock()

        # Asynchronous Background Worker
        self._pending_frame: np.ndarray | None = None
        self._has_frame = threading.Event()
        self._running = True

        _ensure_torch_nms_patch()
        self._load_model()

        # Start async worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path if self.model_path else "yolov8n.pt")
        except ImportError:
            print("[PhoneDetector] Notice: 'ultralytics' not installed. Phone detection offline.")
            self._model = None
        except Exception as e:
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolov8n.pt")
            except Exception:
                self._model = None

    def _worker_loop(self) -> None:
        """Continuous background inference worker."""
        while self._running:
            self._has_frame.wait(timeout=0.1)
            self._has_frame.clear()

            frame_to_process = None
            with self._lock:
                if self._pending_frame is not None:
                    frame_to_process = self._pending_frame
                    self._pending_frame = None

            if frame_to_process is not None:
                self._run_inference(frame_to_process)

    def _validate_geometry(
        self, box: list[int], frame_w: int, frame_h: int
    ) -> bool:
        """Validate bounding box is reasonable size within camera view."""
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Minimum physical pixel constraints (prevents narrow fingers / earbuds)
        min_dim = getattr(config, "PHONE_MIN_DIMENSION_PX", 40)
        min_longest = getattr(config, "PHONE_MIN_LONGEST_SIDE_PX", 70)
        if min(bw, bh) < min_dim or max(bw, bh) < min_longest:
            return False

        aspect_ratio = max(bw / bh, bh / bw)
        if not (config.PHONE_MIN_ASPECT_RATIO <= aspect_ratio <= config.PHONE_MAX_ASPECT_RATIO):
            return False

        box_area = bw * bh
        frame_area = frame_w * frame_h
        area_ratio = box_area / frame_area

        if not (config.PHONE_MIN_AREA_RATIO <= area_ratio <= config.PHONE_MAX_AREA_RATIO):
            return False

        return True

    @staticmethod
    def _is_bare_skin_or_hand(roi_bgr: np.ndarray, max_skin_ratio: float = 0.45) -> bool:
        """
        Calculates skin pixel coverage inside the bounding box.
        If more than max_skin_ratio is bare human skin, it is fingers/palm/arm, NOT a phone.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return False

        # YCrCb color space skin mask
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_ycrcb = (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 130)

        # HSV color space skin mask
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        skin_hsv = ((h <= 25) | (h >= 170)) & (s >= 25) & (s <= 210) & (v >= 40)

        skin_mask = skin_ycrcb & skin_hsv
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(roi_bgr.shape[0] * roi_bgr.shape[1])

        return skin_ratio > max_skin_ratio

    def _run_inference(self, frame_bgr: np.ndarray) -> None:
        """Execute YOLO inference in background thread."""
        if self._model is None:
            return

        fh, fw = frame_bgr.shape[:2]
        raw_phone_detections = []
        remote_boxes = []

        effective_conf = getattr(config, "PHONE_DETECTION_CONFIDENCE", self.conf_threshold)

        try:
            # Run YOLO with sufficient resolution (480px) and clean confidence threshold
            results = self._model(
                frame_bgr,
                imgsz=480,
                conf=max(0.40, effective_conf),
                classes=[config.REMOTE_CLASS_ID, config.PHONE_CLASS_ID],
                verbose=False,
            )

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    box_coords = [x1, y1, x2, y2]

                    if cls_id == config.REMOTE_CLASS_ID and conf >= getattr(config, "REMOTE_SUPPRESSION_CONFIDENCE", 0.50):
                        remote_boxes.append({"box": box_coords, "conf": conf})

                    elif cls_id == config.PHONE_CLASS_ID and conf >= effective_conf:
                        if self._validate_geometry(box_coords, fw, fh):
                            # Crop bounding box and reject if it consists of bare fingers/hand/skin
                            x1_c = max(0, min(fw - 1, x1))
                            y1_c = max(0, min(fh - 1, y1))
                            x2_c = max(x1_c + 1, min(fw, x2))
                            y2_c = max(y1_c + 1, min(fh, y2))
                            roi = frame_bgr[y1_c:y2_c, x1_c:x2_c]

                            max_skin = getattr(config, "PHONE_MAX_SKIN_RATIO", 0.45)
                            if not self._is_bare_skin_or_hand(roi, max_skin_ratio=max_skin):
                                raw_phone_detections.append({
                                    "box": box_coords,
                                    "conf": round(conf, 2),
                                    "class_name": config.PHONE_CLASS_NAME,
                                })

            filtered_phone_detections = []
            for p_det in raw_phone_detections:
                px1, py1, px2, py2 = p_det["box"]
                is_remote_overlap = False

                for r_det in remote_boxes:
                    rx1, ry1, rx2, ry2 = r_det["box"]
                    ix1 = max(px1, rx1)
                    iy1 = max(py1, ry1)
                    ix2 = min(px2, rx2)
                    iy2 = min(py2, ry2)

                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        p_area = (px2 - px1) * (py2 - py1)
                        if (inter_area / p_area > 0.60) and (r_det["conf"] > p_det["conf"] + 0.15):
                            is_remote_overlap = True
                            break

                if not is_remote_overlap:
                    filtered_phone_detections.append(p_det)

        except Exception as e:
            if config.DEBUG_LOGGING:
                print(f"[PhoneDetector] Inference error: {e}")
            filtered_phone_detections = []

        raw_detected = len(filtered_phone_detections) > 0
        required_frames = getattr(config, "PHONE_CONFIRMATION_FRAMES", 2)

        with self._lock:
            if raw_detected:
                self._consecutive_detections += 1
                self._consecutive_misses = 0
                self._last_valid_detections = filtered_phone_detections

                if self._consecutive_detections >= required_frames:
                    if not self._is_phone_confirmed and config.DEBUG_LOGGING:
                        conf_val = filtered_phone_detections[0]["conf"] if filtered_phone_detections else self.conf_threshold
                        now_str = time.strftime("%H:%M:%S")
                        print(f"[{now_str}] [PhoneDetector] 📱 Phone detected (confidence={conf_val:.2f})")
                    self._is_phone_confirmed = True
            else:
                self._consecutive_misses += 1
                self._consecutive_detections = 0

                if self._consecutive_misses >= self.lost_frames:
                    if self._is_phone_confirmed and config.DEBUG_LOGGING:
                        now_str = time.strftime("%H:%M:%S")
                        print(f"[{now_str}] [PhoneDetector] Phone removed")
                    self._is_phone_confirmed = False
                    self._last_valid_detections = []

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """
        Pass frame to background thread and return latest smoothed state immediately (0ms blocking).
        """
        with self._lock:
            self._pending_frame = frame_bgr
            is_confirmed = self._is_phone_confirmed
            dets = list(self._last_valid_detections) if is_confirmed else []
            raw = self._consecutive_detections > 0

        self._has_frame.set()

        return {
            "phone_detected": is_confirmed,
            "detections": dets,
            "raw_detected": raw,
        }

    def close(self) -> None:
        """Stop background worker thread."""
        self._running = False
        self._has_frame.set()

    def draw_detections(
        self, frame_bgr: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        """Render modern cyberpunk AI bounding box with corner brackets."""
        annotated = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["conf"]

            color = (40, 50, 235)  # Vibrant Crimson Red (BGR)
            corner_color = (80, 220, 255)  # Cyan corner accents

            # Main bounding rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # High-tech corner brackets
            line_len = min(20, max(8, (x2 - x1) // 5), max(8, (y2 - y1) // 5))
            thick = 3
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + line_len, y1), corner_color, thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + line_len), corner_color, thick)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - line_len, y1), corner_color, thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + line_len), corner_color, thick)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + line_len, y2), corner_color, thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - line_len), corner_color, thick)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - line_len, y2), corner_color, thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - line_len), corner_color, thick)

            # Pill Badge
            label = f"PHONE {int(conf * 100)}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            badge_y = max(y1 - 10, h + 10)
            cv2.rectangle(annotated, (x1, badge_y - h - 8), (x1 + w + 16, badge_y + 4), (15, 15, 25), -1)
            cv2.rectangle(annotated, (x1, badge_y - h - 8), (x1 + w + 16, badge_y + 4), color, 1)
            cv2.putText(
                annotated,
                label,
                (x1 + 8, badge_y - 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (240, 240, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

