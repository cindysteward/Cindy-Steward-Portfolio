"""
Video preprocessing for the FACET-CV pipeline.

Applies the MediaPipe FaceLandmarker (Tasks API) to video frames and
produces three structured DataFrames per session:

  - frame_df: One row per frame with detection metadata (detection_success,
    confidence, segment, repetition, task annotation, brightness, PSNR,
    occlusion flag).
  - landmarks_df: Wide-format x/y/z coordinates for all 468 face landmarks.
  - blendshapes_df: Per-frame blendshape activation scores with quality columns.

PSNR is estimated every 30 frames using the frame's own median-blurred version
as a noise-free reference and held constant for the intervening frames.

Reference: Lugaresi et al. (2019) MediaPipe: A framework for building
perception pipelines. CVPR Workshop on Computer Vision for AR/VR.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .utils import ensure_model_downloaded


@dataclass
class FrameResult:
    """Per-frame detection results from the FaceLandmarker.

    Attributes:
        frame_index: Zero-based frame number within the video.
        timestamp_abs: Absolute timestamp in seconds from the recording start.
        detection_success: True when at least one face was detected.
        detection_confidence: Detection confidence score (0.0-1.0).
        landmarks: Array of shape (468, 3) with pixel-space x/y/z coordinates,
            or None when detection failed.
        blendshapes: Dict mapping blendshape name to activation score,
            or None when detection failed.
        segment: Segment type label ('neutral', 'measurement', or None).
        repetition: Repetition number within the current task segment.
        task_group: Section identifier ('A', 'B', 'C', or '0' for baseline).
        task_id: Numeric task identifier within the section.
        task_name: Human-readable task label from the study prompter.
        brightness: Mean pixel brightness (0-255) computed from a 4x subsampled
            version of the frame for speed.
    """
    frame_index: int
    timestamp_abs: float
    detection_success: bool
    detection_confidence: float
    landmarks: Optional[np.ndarray]
    blendshapes: Optional[Dict[str, float]]
    segment: Optional[str]
    repetition: int
    task_group: Optional[str] = None
    task_id: Optional[int] = None
    task_name: Optional[str] = None
    brightness: Optional[float] = None


class Preprocessor:
    """Runs MediaPipe FaceLandmarker on video frames and produces structured DataFrames."""

    def __init__(self, features_config: Dict[str, Any]):
        """Initialise the FaceLandmarker and collect configured blendshape names.

        Downloads the model file if it is not already present (see
        utils.ensure_model_downloaded).  All MediaPipe settings (max_num_faces,
        confidence thresholds, blendshape output) are read from the
        mediapipe_settings section of features_config.

        Args:
            features_config: Dict loaded from features.yaml.
        """
        self.features_config = features_config
        mp_settings = features_config.get("mediapipe_settings", {})

        model_path = ensure_model_downloaded()
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=mp_settings.get("max_num_faces", 1),
            min_face_detection_confidence=mp_settings.get("min_detection_confidence", 0.5),
            min_face_presence_confidence=mp_settings.get("min_tracking_confidence", 0.5),
            min_tracking_confidence=mp_settings.get("min_tracking_confidence", 0.5),
            output_face_blendshapes=mp_settings.get("output_face_blendshapes", True),
            output_facial_transformation_matrixes=False,
        )

        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.blendshape_names = self._get_all_blendshape_names()

    def _get_all_blendshape_names(self) -> List[str]:
        """Collect all configured blendshape names across facial regions."""
        names: List[str] = []
        for region_blendshapes in self.features_config.get("blendshapes", {}).values():
            if isinstance(region_blendshapes, list):
                names.extend(region_blendshapes)
        return names

    def process_video(
        self, video_path: Path, frame_data: List[Dict], events_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process an entire video through the FaceLandmarker.

        Reads every frame with OpenCV, applies optional brightness normalisation,
        runs face landmark detection, and appends PSNR every 30 frames.

        Args:
            video_path: Path to the MP4 video file.
            frame_data: List of per-frame dicts from the study prompter pipeline,
                keyed by frame_index, supplying timestamp_abs, segment, repetition,
                task_group, task_id, and task_name.
            events_df: Not used directly in this method; reserved for callers
                that need to pass it through.

        Returns:
            Tuple of (frame_df, landmarks_df, blendshapes_df).

        Raises:
            RuntimeError: When the video file cannot be opened.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frame_lookup = {fd["frame_index"]: fd for fd in frame_data}
        results_list: List[FrameResult] = []
        psnr_values: List[float] = []
        _PSNR_STRIDE = 30
        _last_psnr: float = float("nan")
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fd = frame_lookup.get(frame_index, {})
            timestamp = fd.get("timestamp_abs", frame_index / 30.0)
            segment = fd.get("segment")
            repetition = fd.get("repetition", 0)
            task_group = fd.get("task_group")
            task_id = fd.get("task_id")
            task_name = fd.get("task_name")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            bn_cfg = self.features_config.get("brightness_normalization", {})
            if bn_cfg.get("enabled", False):
                target = float(bn_cfg.get("target_mean", 128.0))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                v = hsv[:, :, 2]
                cur_mean = np.mean(v) if v.size > 0 else target
                if cur_mean > 0:
                    v = np.clip(v * (target / cur_mean), 0, 255)
                    hsv[:, :, 2] = v
                    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            brightness = float(frame[::4, ::4].mean())
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.face_landmarker.detect(mp_image)

            result = self._process_frame_results(
                detection_result, frame_index, timestamp, segment, repetition,
                frame.shape, task_group=task_group, task_id=task_id,
                task_name=task_name, brightness=brightness,
            )
            results_list.append(result)
            if frame_index % _PSNR_STRIDE == 0:
                _last_psnr = self.compute_psnr(frame)
            psnr_values.append(_last_psnr)
            frame_index += 1

        cap.release()

        frame_df = self._create_frame_dataframe(results_list)
        landmarks_df = self._create_landmarks_dataframe(results_list)
        blendshapes_df = self._create_blendshapes_dataframe(results_list)

        while len(psnr_values) < len(results_list):
            psnr_values.append(float("nan"))
        frame_df["psnr"] = psnr_values[: len(frame_df)]
        return frame_df, landmarks_df, blendshapes_df

    def process_frame(
        self, frame: np.ndarray, frame_index: int, timestamp: float,
        segment: Optional[str] = None, repetition: int = 0,
    ) -> FrameResult:
        """Run face landmark detection on a single BGR frame.

        Converts to RGB, runs the FaceLandmarker, and returns a FrameResult.
        Used by callers that supply frames one at a time rather than from a video.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_landmarker.detect(mp_image)
        brightness = float(frame[::4, ::4].mean())
        return self._process_frame_results(
            detection_result, frame_index, timestamp, segment, repetition,
            frame.shape, brightness=brightness,
        )

    def _process_frame_results(
        self, detection_result, frame_index: int, timestamp: float,
        segment: Optional[str], repetition: int, frame_shape: Tuple[int, ...],
        task_group: Optional[str] = None, task_id: Optional[int] = None,
        task_name: Optional[str] = None, brightness: Optional[float] = None,
    ) -> FrameResult:
        """Convert raw MediaPipe detection output into a FrameResult dataclass.

        On detection failure, returns a FrameResult with detection_success=False
        and None for landmarks and blendshapes.  On success, pixel-space landmark
        coordinates are computed by multiplying the normalised x/y values by
        frame width/height.
        """
        if not detection_result.face_landmarks:
            return FrameResult(
                frame_index=frame_index, timestamp_abs=timestamp,
                detection_success=False, detection_confidence=0.0,
                landmarks=None, blendshapes=None,
                segment=segment, repetition=repetition,
                task_group=task_group, task_id=task_id,
                task_name=task_name, brightness=brightness,
            )

        face_landmarks = detection_result.face_landmarks[0]
        h, w = frame_shape[:2]
        raw_lm = [[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks]
        landmarks = np.array(raw_lm[:468]) if raw_lm else np.empty((0, 3), dtype=float)

        blendshapes: Dict[str, float] = {}
        if detection_result.face_blendshapes:
            for bs in detection_result.face_blendshapes[0]:
                blendshapes[bs.category_name] = bs.score

        return FrameResult(
            frame_index=frame_index, timestamp_abs=timestamp,
            detection_success=True, detection_confidence=1.0,
            landmarks=landmarks, blendshapes=blendshapes,
            segment=segment, repetition=repetition,
            task_group=task_group, task_id=task_id,
            task_name=task_name, brightness=brightness,
        )

    def _create_frame_dataframe(self, results: List[FrameResult]) -> pd.DataFrame:
        """Build a per-frame DataFrame with detection metadata and task annotations."""
        occlusion_thresh = float(self.features_config.get("occlusion_confidence_thresh", 0.5))
        rows = []
        for r in results:
            rows.append({
                "frame_index": r.frame_index,
                "timestamp_abs": r.timestamp_abs,
                "detection_success": r.detection_success,
                "detection_confidence": r.detection_confidence,
                "segment": r.segment,
                "repetition": r.repetition,
                "task_group": r.task_group,
                "task_id": r.task_id,
                "task_name": r.task_name,
                "brightness": r.brightness,
                "occluded": (not r.detection_success) or (r.detection_confidence < occlusion_thresh),
            })
        return pd.DataFrame(rows)

    def _create_landmarks_dataframe(self, results: List[FrameResult]) -> pd.DataFrame:
        """Build a wide-format DataFrame of per-landmark x/y/z coordinates.

        Columns are named lm_{i}_x, lm_{i}_y, lm_{i}_z for landmark index i.
        Rows for frames with no detection have no landmark columns populated.
        """
        rows = []
        for r in results:
            row: Dict[str, Any] = {
                "frame_index": r.frame_index,
                "timestamp_abs": r.timestamp_abs,
                "segment": r.segment,
                "repetition": r.repetition,
            }
            if r.landmarks is not None:
                for i, (x, y, z) in enumerate(r.landmarks):
                    row[f"lm_{i}_x"] = x
                    row[f"lm_{i}_y"] = y
                    row[f"lm_{i}_z"] = z
            rows.append(row)
        return pd.DataFrame(rows)

    def _create_blendshapes_dataframe(self, results: List[FrameResult]) -> pd.DataFrame:
        """Build a DataFrame of per-frame blendshape activation scores with quality metadata.

        Includes inter_ocular_distance (from landmarks 33 and 263), brightness,
        and an occluded flag.  Blendshape columns are absent for failed-detection
        rows.
        """
        occlusion_thresh = float(self.features_config.get("occlusion_confidence_thresh", 0.5))
        rows = []
        for r in results:
            row: Dict[str, Any] = {
                "frame_index": r.frame_index,
                "timestamp_abs": r.timestamp_abs,
                "segment": r.segment,
                "repetition": r.repetition,
                "detection_success": r.detection_success,
            }
            if r.blendshapes:
                row.update(r.blendshapes)

            if r.landmarks is not None and len(r.landmarks) > 263 and len(r.landmarks) > 33:
                left_eye = r.landmarks[33]
                right_eye = r.landmarks[263]
                row["inter_ocular_distance"] = float(np.linalg.norm(left_eye[:2] - right_eye[:2]))
            else:
                row["inter_ocular_distance"] = np.nan

            row["brightness"] = r.brightness
            row["occluded"] = (not r.detection_success) or (r.detection_confidence < occlusion_thresh)
            rows.append(row)

        return pd.DataFrame(rows)

    def extract_segment_data(self, blendshapes_df: pd.DataFrame, segment_type: str) -> pd.DataFrame:
        """Return a copy of rows whose 'segment' column matches segment_type."""
        return blendshapes_df[blendshapes_df["segment"] == segment_type].copy()

    def get_detection_quality_stats(self, frame_df: pd.DataFrame) -> Dict[str, float]:
        """Compute detection-quality summary statistics over a frame DataFrame."""
        total_frames = len(frame_df)
        detected_frames = int(frame_df["detection_success"].sum())
        return {
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate": detected_frames / total_frames if total_frames > 0 else 0.0,
            "mean_confidence": float(frame_df["detection_confidence"].mean()),
            "mean_psnr": float(frame_df["psnr"].mean()) if "psnr" in frame_df.columns else float("nan"),
            "min_psnr": float(frame_df["psnr"].min()) if "psnr" in frame_df.columns else float("nan"),
        }

    def close(self) -> None:
        """Release the FaceLandmarker resources."""
        if hasattr(self, "face_landmarker"):
            self.face_landmarker.close()

    @staticmethod
    def compute_psnr(frame: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
        """Compute PSNR using a median-blurred reference when none is given."""
        if reference is None:
            reference = cv2.medianBlur(frame, 3)
        mse = np.mean((frame.astype(np.float32) - reference.astype(np.float32)) ** 2)
        if mse == 0:
            return 100.0
        return float(20 * np.log10(255.0 / np.sqrt(mse)))


def create_preprocessor(features_config: Dict[str, Any]) -> Preprocessor:
    """Factory: build a Preprocessor from features configuration."""
    return Preprocessor(features_config)
