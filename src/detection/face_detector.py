"""
Google MediaPipe Face & Landmark Detector (Task API)
===================================================

Uses Google's official `face_landmarker.task` 478-point 3D Face Mesh model.
Provides facial landmark coordinates, face blendshapes, and multi-face counting.
"""

from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import config


class FaceMeshDetector:
    """
    Google MediaPipe 478-point 3D Face Landmark Detector.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or str(config.MODELS_DIR / "face_landmarker.task")
        self.detector = None
        self._load_model()

    def _load_model(self) -> None:
        """Initialize Google MediaPipe FaceLandmarker."""
        task_path = Path(self.model_path)
        if not task_path.exists():
            # Download if missing
            import urllib.request
            task_path.parent.mkdir(parents=True, exist_ok=True)
            print("[FaceMeshDetector] Downloading official Google face_landmarker.task...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                str(task_path),
            )

        try:
            base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=2,
                min_face_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                min_face_presence_confidence=config.FACE_TRACKING_CONFIDENCE,
                min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("[FaceMeshDetector] Google MediaPipe FaceLandmarker loaded successfully.")
        except Exception as e:
            print(f"[FaceMeshDetector] Critical error initializing FaceLandmarker: {e}")
            self.detector = None

    def process(self, frame_bgr: np.ndarray):
        """
        Process a BGR frame.

        Returns
        -------
        results_dict, face_count
        """
        if self.detector is None:
            return {"landmarks": [], "blendshapes": []}, 0

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            detection_result = self.detector.detect(mp_image)
            face_count = len(detection_result.face_landmarks) if detection_result.face_landmarks else 0
            return {
                "face_landmarks": detection_result.face_landmarks,
                "face_blendshapes": detection_result.face_blendshapes,
            }, face_count
        except Exception as e:
            return {"face_landmarks": [], "face_blendshapes": []}, 0

    def close(self) -> None:
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
