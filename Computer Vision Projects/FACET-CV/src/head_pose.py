"""
Head pose estimation and normalisation for the FACET-CV pipeline.

Estimates 3D head orientation (pitch, yaw, roll in degrees) from a small set
of named facial landmarks and provides tools to correct asymmetry features for
head pose variation between sessions.

The geometric approach uses the nose-tip position relative to eye and mouth
centres for pitch/yaw, and the eye-corner angle for roll.  This is sufficient
for the moderate pose ranges encountered in clinical recordings (both upright
and supine) and is validated by Hammami et al. (2022).

Supine-position correction rationale
--------------------------------------
MediaPipe FaceLandmarker blendshapes are computed relative to the detected
face orientation and are therefore largely invariant to global head pose.
Asymmetry metrics derived from blendshape ratios should remain valid across
upright and supine positions.

Gravity shifts soft tissue toward the ears in the supine position.  This is
a real physiological change but should not be interpreted as pathological
asymmetry when the reference session was recorded upright.  The pipeline
handles this in three ways:

1. Within-session neutral baseline: asymmetry is expressed relative to the
   subject's own neutral captured at the start of that session, so
   session-internal comparisons are pose-agnostic.

2. Pitch-deviation flag: when pitch deviation from the reference pose exceeds
   SUPINE_PITCH_THRESHOLD degrees, a session-level flag is set so downstream
   cross-session comparisons can apply a caveat or exclusion rule.

3. Roll compensation: for lateral head tilt, a linear correction factor
   (ROLL_ASYMMETRY_CORRECTION_FACTOR) is applied to left/right asymmetry ratios.

References
----------
Hammami M, Ghazouani H, Farah IR (2022) Evaluation of various state of the
art head pose estimation algorithms for clinical scenarios. Sensors 22(18):6850.
doi:10.3390/s22186850

Lugaresi et al. (2019) MediaPipe: A framework for building perception pipelines.
CVPR Workshop on Computer Vision for AR/VR.
"""

import numpy as np
from typing import Dict, Tuple, Optional

SUPINE_PITCH_THRESHOLD: float = 40.0
ROLL_ASYMMETRY_CORRECTION_FACTOR: float = 0.08


class HeadPoseNormalizer:
    """Estimate head pose from facial landmarks and correct asymmetry for pose changes.

    Typical usage:
      1. Call estimate_pose() on each frame to get yaw/pitch/roll.
      2. Call set_reference_pose() with the median pose from the neutral baseline.
      3. Call compute_pose_deviation() to check how far the current frame is from
         the reference.
      4. Call correct_asymmetry_features() to apply roll correction.
    """

    def __init__(self):
        """Initialise with no stored reference pose."""
        self.reference_pose: Optional[Dict[str, float]] = None

    def estimate_pose(self, landmarks_dict: Dict) -> Dict[str, float]:
        """Estimate head pose from named facial landmarks and return yaw, pitch, roll in degrees.

        Uses nose-tip position relative to the eye midpoint for yaw, nose-tip
        vertical offset relative to the eye-mouth axis for pitch, and the
        eye-corner slope for roll.  All angles are expressed as deviations from
        a camera-facing neutral position.

        Args:
            landmarks_dict: Dict with keys 'noseTip', 'leftEye', 'rightEye',
                'mouthLeft', 'mouthRight', each mapping to a 3-element list
                [x, y, z] in normalised image coordinates.

        Returns:
            Dict with float keys 'yaw', 'pitch', 'roll' in degrees.
        """
        nose_tip = landmarks_dict.get('noseTip', [0, 0, 0])
        left_eye = landmarks_dict.get('leftEye', [0, 0, 0])
        right_eye = landmarks_dict.get('rightEye', [0, 0, 0])
        left_mouth = landmarks_dict.get('mouthLeft', [0, 0, 0])
        right_mouth = landmarks_dict.get('mouthRight', [0, 0, 0])

        nose = np.array(nose_tip[:3])
        left_e = np.array(left_eye[:3])
        right_e = np.array(right_eye[:3])
        left_m = np.array(left_mouth[:3])
        right_m = np.array(right_mouth[:3])

        eye_center = (left_e + right_e) / 2
        mouth_center = (left_m + right_m) / 2

        eye_span  = np.abs(right_e[0] - left_e[0])
        nose_off_x = nose[0] - (right_e[0] + left_e[0]) / 2
        half_span  = max(eye_span / 2, 1e-6)
        yaw = float(np.degrees(np.arctan2(nose_off_x, half_span)))

        vertical_line = mouth_center[1] - eye_center[1]
        nose_offset   = nose[1] - eye_center[1]
        pitch = np.arctan2(nose_offset - vertical_line / 2,
                           max(abs(vertical_line), 1e-6)) * 180 / np.pi

        eye_dy = right_e[1] - left_e[1]
        roll   = np.arctan2(eye_dy, max(eye_span, 1e-6)) * 180 / np.pi

        return {
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll),
        }

    def set_reference_pose(self, pose: Dict[str, float]):
        """Store a reference head pose to use as the baseline for deviation calculations."""
        self.reference_pose = pose

    def compute_pose_deviation(self, current_pose: Dict[str, float]) -> float:
        """Compute total angular deviation from reference pose in degrees.

        Returns 0.0 when no reference has been stored.
        """
        if self.reference_pose is None:
            return 0.0

        d_yaw = current_pose['yaw'] - self.reference_pose['yaw']
        d_pitch = current_pose['pitch'] - self.reference_pose['pitch']
        d_roll = current_pose['roll'] - self.reference_pose['roll']

        deviation = np.sqrt(d_yaw ** 2 + d_pitch ** 2 + d_roll ** 2)
        return float(deviation)

    def is_supine(self, current_pose: Dict[str, float]) -> bool:
        """Return True when the pitch deviation from reference exceeds the supine threshold.

        A pitch deviation larger than SUPINE_PITCH_THRESHOLD degrees indicates
        the subject is likely lying down (supine or close to horizontal).
        Within-session asymmetry measurements are still valid because they are
        computed relative to the within-session neutral baseline.  Cross-session
        comparisons should flag this condition.
        """
        if self.reference_pose is None:
            return False
        d_pitch = abs(current_pose['pitch'] - self.reference_pose['pitch'])
        return d_pitch > SUPINE_PITCH_THRESHOLD

    def should_warn_pose(self, deviation: float) -> Tuple[bool, str]:
        """Return (should_warn, message) for a given pose deviation in degrees.

        Returns (True, warning_string) when deviation exceeds 10 degrees, with
        a stronger message above 15 degrees.  Returns (False, '') otherwise.
        """
        if deviation > 15.0:
            return True, (
                "HEAD POSE WARNING: Head position differs significantly "
                "from baseline. Please face camera directly."
            )
        elif deviation > 10.0:
            return True, (
                "Moderate head pose variation detected. "
                "Results may be less reliable."
            )
        else:
            return False, ""

    def correct_asymmetry_features(
        self,
        features: Dict[str, float],
        current_pose: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply a linear roll correction to all asymmetry features.

        Corrects left/right asymmetry values for lateral head tilt by
        subtracting a term proportional to the roll deviation from the
        reference pose.  The correction factor is ROLL_ASYMMETRY_CORRECTION_FACTOR.

        Pitch-induced tissue shift is handled upstream by using the within-session
        neutral baseline rather than a direct numerical correction, because the
        direction and magnitude of gravity-induced soft-tissue displacement varies
        non-linearly with tissue compliance.

        Returns features unchanged when no reference pose has been set.
        """
        if self.reference_pose is None:
            return features

        corrected = features.copy()
        roll_diff = current_pose['roll'] - self.reference_pose['roll']

        for key, value in features.items():
            if 'asymmetry' in key.lower():
                correction_factor = -ROLL_ASYMMETRY_CORRECTION_FACTOR * roll_diff
                corrected[key] = value + correction_factor

        return corrected
