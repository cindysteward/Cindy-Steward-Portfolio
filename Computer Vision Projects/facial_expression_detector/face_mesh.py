"""
Face detection and landmark extraction using MediaPipe Face Mesh.
Provides real-time face detection and 478 landmark coordinates for expression analysis.
"""

from typing import Tuple, List
import mediapipe as mp
import numpy as np


class FaceLandmarks:
    """Index constants for key facial landmark groups in MediaPipe Face Mesh."""

    LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 246]
    RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466]
    MOUTH = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181,
    ]
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]
    FACE_CONTOUR = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]


class FaceMeshDetector:
    """Wraps MediaPipe Face Mesh for real-time face detection and landmark extraction."""

    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def detect_face(self, frame: np.ndarray) -> Tuple[bool, List]:
        """Detect a face and return (found, landmark_list) with normalized coordinates."""
        rgb_frame = frame[:, :, ::-1]
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            landmark_list = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
            return True, landmark_list
        return False, []

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
