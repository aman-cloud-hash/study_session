"""
Eye State Detector (Google MediaPipe 478-Point EAR & Blendshape Analysis)
========================================================================

Extracts exact 3D eye landmarks and Google FaceBlendshapes (`eyeBlinkLeft`, `eyeBlinkRight`)
to classify eye openness state as OPEN or CLOSED.
"""

import cv2
import numpy as np
import config


class EyeDetector:
    """
    Precision Eye State & EAR Detector using MediaPipe FaceMesh landmarks.
    """

    # Exact 6-point MediaPipe eye contour indices
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

    # Full eye contours for visual rendering
    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def __init__(self, ear_threshold: float = config.EAR_THRESHOLD) -> None:
        self.ear_threshold = ear_threshold

    @staticmethod
    def _euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _calculate_single_eye_ear(
        self, landmarks, eye_indices: list[int], img_w: int, img_h: int
    ) -> tuple[float, list[tuple[int, int]]]:
        coords = []
        for idx in eye_indices:
            lm = landmarks[idx]
            coords.append(np.array([lm.x * img_w, lm.y * img_h]))

        p1, p2, p3, p4, p5, p6 = coords
        v1 = self._euclidean_distance(p2, p6)
        v2 = self._euclidean_distance(p3, p5)
        h = self._euclidean_distance(p1, p4)

        if h < 1e-6:
            return 0.0, [(int(c[0]), int(c[1])) for c in coords]

        ear = (v1 + v2) / (2.0 * h)
        pixel_points = [(int(c[0]), int(c[1])) for c in coords]
        return ear, pixel_points

    def detect_eye_state_from_face(
        self, frame_bgr: np.ndarray, results_dict: dict
    ) -> dict:
        """
        Calculates EAR and eye status from MediaPipe 478 3D landmarks & blendshapes.
        """
        h, w = frame_bgr.shape[:2]
        face_landmarks_list = results_dict.get("face_landmarks", [])

        if not face_landmarks_list or len(face_landmarks_list) == 0:
            return {
                "avg_ear": 0.0,
                "eye_status": "UNKNOWN",
                "left_points": [],
                "right_points": [],
                "blink_score": 0.0,
            }

        # Primary face
        primary_face = face_landmarks_list[0]

        # 1. 6-point 3D EAR calculation
        l_ear, l_pts = self._calculate_single_eye_ear(primary_face, self.LEFT_EYE_LANDMARKS, w, h)
        r_ear, r_pts = self._calculate_single_eye_ear(primary_face, self.RIGHT_EYE_LANDMARKS, w, h)
        avg_ear = (l_ear + r_ear) / 2.0

        # 2. MediaPipe Blendshapes (eyeBlinkLeft, eyeBlinkRight)
        blink_left = 0.0
        blink_right = 0.0
        blendshapes_list = results_dict.get("face_blendshapes", [])
        if blendshapes_list and len(blendshapes_list) > 0:
            for category in blendshapes_list[0]:
                if category.category_name == "eyeBlinkLeft":
                    blink_left = category.score
                elif category.category_name == "eyeBlinkRight":
                    blink_right = category.score

        avg_blink = (blink_left + blink_right) / 2.0

        # 3. Dual-Validation Classification Rule
        # Eyes are CLOSED if EAR drops below threshold OR blendshape blink score is high (> 0.50)
        is_closed = (avg_ear < self.ear_threshold) or (avg_blink > 0.50)
        eye_status = "CLOSED" if is_closed else "OPEN"

        # Extract full eye contours for rendering
        l_contour = [(int(primary_face[idx].x * w), int(primary_face[idx].y * h)) for idx in self.LEFT_EYE_CONTOUR]
        r_contour = [(int(primary_face[idx].x * w), int(primary_face[idx].y * h)) for idx in self.RIGHT_EYE_CONTOUR]

        return {
            "avg_ear": round(avg_ear, 2),
            "eye_status": eye_status,
            "left_points": l_contour,
            "right_points": r_contour,
            "blink_score": round(avg_blink, 2),
        }

    def draw_eye_annotations(
        self, frame_bgr: np.ndarray, eye_data: dict
    ) -> np.ndarray:
        """
        Passes through clean frame without drawing over student's eyes.
        Eye state and drowsiness detection continue running in the background.
        """
        return frame_bgr
