"""
Facial expression detection using calibrated landmark-based heuristics.
Computes normalized facial measurements from MediaPipe landmarks and compares
against a per-user baseline to determine whether a target expression is being performed.
"""

from typing import List, Tuple, Optional, Dict
from utils import distance, RollingAverage, SMOOTHING_WINDOW, CALIBRATION_FRAMES


EXPRESSIONS = [
    "Smile",
    "Big Grin",
    "Left Wink",
    "Right Wink",
    "Raised Eyebrows",
    "Kiss Face",
    "Squint",
    "Surprised",
    "Wide Eyes",
    "Jaw Drop",
    "Tongue Out",
]


class FaceMeasurements:
    """Normalized facial measurements derived from raw face mesh landmarks."""

    def __init__(self, landmarks: List[Tuple[float, float, float]]):
        left_eye = landmarks[33][:2]
        right_eye = landmarks[263][:2]
        self.eye_dist = distance(left_eye, right_eye)

        if self.eye_dist <= 0:
            self.valid = False
            return
        self.valid = True

        self.mouth_width = distance(landmarks[61][:2], landmarks[291][:2]) / self.eye_dist
        self.mouth_open = distance(landmarks[13][:2], landmarks[14][:2]) / self.eye_dist

        self.left_eye_open = distance(landmarks[159][:2], landmarks[145][:2]) / self.eye_dist
        self.right_eye_open = distance(landmarks[386][:2], landmarks[374][:2]) / self.eye_dist

        left_brow_y = landmarks[70][1]
        right_brow_y = landmarks[300][1]
        left_eye_y = (landmarks[159][1] + landmarks[145][1]) / 2
        right_eye_y = (landmarks[386][1] + landmarks[374][1]) / 2
        self.left_brow_height = (left_eye_y - left_brow_y) / self.eye_dist
        self.right_brow_height = (right_eye_y - right_brow_y) / self.eye_dist
        self.avg_brow_height = (self.left_brow_height + self.right_brow_height) / 2

        mouth_corner_avg_y = (landmarks[61][1] + landmarks[291][1]) / 2
        lip_center_y = (landmarks[13][1] + landmarks[14][1]) / 2
        self.mouth_upturn = (mouth_corner_avg_y - lip_center_y) / self.eye_dist

        self.mouth_stretch = distance(landmarks[0][:2], landmarks[17][:2]) / self.eye_dist

        self.cheek_width = distance(landmarks[234][:2], landmarks[454][:2]) / self.eye_dist


class ExpressionDetector:
    """Detects specific facial expressions relative to a per-user calibrated baseline."""

    def __init__(self):
        self.baseline: Optional[Dict[str, float]] = None
        self.calibrating = True
        self._cal_values: List[FaceMeasurements] = []
        self.smoothing = {expr: RollingAverage(SMOOTHING_WINDOW) for expr in EXPRESSIONS}

    @property
    def calibration_progress(self) -> float:
        """Return calibration completion as a value from 0.0 to 1.0."""
        return min(len(self._cal_values) / CALIBRATION_FRAMES, 1.0)

    def update(self, landmarks: List, target_expression: str) -> Tuple[bool, bool]:
        """Process a frame's landmarks and return (calibrated, expression_detected)."""
        if not landmarks or len(landmarks) < 468:
            return (not self.calibrating, False)

        m = FaceMeasurements(landmarks)
        if not m.valid:
            return (not self.calibrating, False)

        if self.calibrating:
            self._cal_values.append(m)
            if len(self._cal_values) >= CALIBRATION_FRAMES:
                self._compute_baseline()
                self.calibrating = False
            return (False, False)

        return (True, self._detect(m, target_expression))

    def _compute_baseline(self):
        """Average all calibration measurements to establish the neutral baseline."""
        vals = self._cal_values
        n = len(vals)
        self.baseline = {
            "mouth_width": sum(v.mouth_width for v in vals) / n,
            "mouth_open": sum(v.mouth_open for v in vals) / n,
            "left_eye_open": sum(v.left_eye_open for v in vals) / n,
            "right_eye_open": sum(v.right_eye_open for v in vals) / n,
            "left_brow_height": sum(v.left_brow_height for v in vals) / n,
            "right_brow_height": sum(v.right_brow_height for v in vals) / n,
            "avg_brow_height": sum(v.avg_brow_height for v in vals) / n,
            "mouth_upturn": sum(v.mouth_upturn for v in vals) / n,
            "mouth_stretch": sum(v.mouth_stretch for v in vals) / n,
            "cheek_width": sum(v.cheek_width for v in vals) / n,
        }

    def _detect(self, m: FaceMeasurements, target: str) -> bool:
        """Run the appropriate detector for the target expression with smoothing."""
        b = self.baseline
        detectors = {
            "Smile": self._smile,
            "Big Grin": self._big_grin,
            "Left Wink": self._left_wink,
            "Right Wink": self._right_wink,
            "Raised Eyebrows": self._raised_eyebrows,
            "Kiss Face": self._kiss_face,
            "Squint": self._squint,
            "Surprised": self._surprised,
            "Wide Eyes": self._wide_eyes,
            "Jaw Drop": self._jaw_drop,
            "Tongue Out": self._tongue_out,
        }

        fn = detectors.get(target)
        if fn is None:
            return False

        raw = 1.0 if fn(m, b) else 0.0
        smoothed = self.smoothing[target].add(raw)
        return smoothed >= 0.5

    def reset_smoothing(self):
        """Clear all smoothing buffers between expression transitions."""
        for avg in self.smoothing.values():
            avg.reset()

    def recalibrate(self):
        """Reset the detector to recalibrate for a new user."""
        self.baseline = None
        self.calibrating = True
        self._cal_values = []
        self.reset_smoothing()

    def _smile(self, m, b):
        """Detect a smile."""
        return (
            m.mouth_width > b["mouth_width"] + 0.025
            and m.mouth_upturn < b["mouth_upturn"] - 0.003
            and m.mouth_open < b["mouth_open"] + 0.07
        )

    def _big_grin(self, m, b):
        """Detect a big grin (wide smile with teeth showing)."""
        return (
            m.mouth_width > b["mouth_width"] + 0.045
            and m.mouth_open > b["mouth_open"] + 0.005
            and m.mouth_upturn < b["mouth_upturn"] - 0.002
        )

    def _left_wink(self, m, b):
        """Detect a left wink."""
        return (
            m.left_eye_open < b["left_eye_open"] * 0.45
            and m.right_eye_open > b["right_eye_open"] * 0.55
        )

    def _right_wink(self, m, b):
        """Detect a right wink."""
        return (
            m.right_eye_open < b["right_eye_open"] * 0.45
            and m.left_eye_open > b["left_eye_open"] * 0.55
        )

    def _raised_eyebrows(self, m, b):
        """Detect raised eyebrows."""
        return (
            m.avg_brow_height > b["avg_brow_height"] + 0.035
            and m.left_brow_height > b["left_brow_height"] + 0.02
            and m.right_brow_height > b["right_brow_height"] + 0.02
        )

    def _pursed_lips(self, m, b):
        """Detect pursed lips."""
        return (
            m.mouth_width < b["mouth_width"] - 0.025
            and m.mouth_open < b["mouth_open"] + 0.015
        )

    def _kiss_face(self, m, b):
        """Detect a kiss face."""
        return (
            m.mouth_width < b["mouth_width"] - 0.02
            and m.mouth_open > b["mouth_open"] + 0.004
            and m.mouth_open < b["mouth_open"] + 0.10
        )

    def _squint(self, m, b):
        """Detect a squint."""
        return (
            m.left_eye_open < b["left_eye_open"] * 0.72
            and m.right_eye_open < b["right_eye_open"] * 0.72
            and m.left_eye_open > b["left_eye_open"] * 0.05
            and m.right_eye_open > b["right_eye_open"] * 0.05
        )

    def _surprised(self, m, b):
        """Detect a surprised face."""
        return (
            m.avg_brow_height > b["avg_brow_height"] + 0.015
            and m.mouth_open > b["mouth_open"] + 0.02
        )

    def _wide_eyes(self, m, b):
        """Detect wide eyes."""
        return (
            m.left_eye_open > b["left_eye_open"] * 1.12
            and m.right_eye_open > b["right_eye_open"] * 1.12
        )

    def _jaw_drop(self, m, b):
        """Detect a jaw drop."""
        return m.mouth_open > b["mouth_open"] + 0.08

    def _tongue_out(self, m, b):
        """Detect tongue sticking out (outer lip stretch increases significantly)."""
        return (
            m.mouth_stretch > b["mouth_stretch"] + 0.04
            and m.mouth_open > b["mouth_open"] + 0.01
            and m.mouth_width < b["mouth_width"] + 0.03
        )
