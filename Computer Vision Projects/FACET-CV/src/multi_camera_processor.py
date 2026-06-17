"""
Multi-camera video processor for FACET-CV study-prompter recordings.

Loads one to four camera video files from a study-prompter session, aligns
them via audio cross-correlation using ffmpeg and scipy, runs MediaPipe
FaceLandmarker on frames from each camera, fuses landmark and blendshape
results across cameras, and produces the frame_data list and events_df that
the pipeline's downstream stages consume.

Camera synchronisation uses one of two strategies, applied in priority order.
When the study-prompter logs per-camera start timestamps, those metadata
offsets are applied directly via apply_offsets_from_meta(). When metadata
offsets are unavailable, align_cameras() extracts the first 30 seconds of
audio from each video using ffmpeg, computes the cross-correlation lag between
each auxiliary camera and the primary camera, and stores the result as a
per-camera time_offset_s.

Landmark fusion across cameras uses a visibility-weighted blending approach.
Blendshapes are weighted by a per-side lateral visibility score derived from
each camera's estimated yaw. Landmark positions are blended per landmark using
a geometric-mean visibility score that combines lateral and vertical camera
geometry, fully vectorised over all 478 MediaPipe landmarks.

References
----------
Kisku DR, Bhatt H, Tistarelli M, Sing JK (2025) Robust multi-camera view
  face recognition. arXiv:1003.05861.
  Multi-view fusion strategy using confidence-weighted landmark evidence;
  motivates the frontality-score weighting in _fuse_landmarks_weighted()
  and the per-camera detection-score weighting scheme.
  https://arxiv.org/pdf/1003.05861

Boström J (2021) Positioning and Tracking using Image Recognition and
  Triangulation. MSc thesis, DIVA-portal.
  Triangulation-based cross-camera 3D position estimation from 2D landmark
  detections; background for the audio cross-correlation alignment and
  multi-camera landmark fusion implemented here.
  https://www.diva-portal.org/smash/get/diva2:1576987/FULLTEXT01.pdf

Frajtag J, Matl M, Simandl M, et al. (2025) Evaluation of facial landmark
  localization performance in a surgical setting. arXiv:2507.18248.
  Benchmark of MediaPipe and competing landmark detectors under operating-room
  lighting and non-frontal head-position conditions; directly informs the
  accepted accuracy envelope and the choice of MediaPipe FaceLandmarker as
  the multi-camera backbone.
  https://arxiv.org/abs/2507.18248

Kitaguchi D, Takeshita N, Matsuzaki H, et al. (2022) Artificial intelligence-
  based computer vision in surgery: recent advances and future perspectives.
  Ann Gastroenterol Surg 6(1):4-14. doi:10.1002/ags3.12513
  Reviews deep learning CV pipelines for intraoperative video analysis
  (detection, segmentation, tracking); contextualises the multi-camera
  processing approach within the broader field of surgical CV.

Lugaresi C, Tang J, Nash H, et al. (2019) MediaPipe: A framework for building
  perception pipelines. arXiv:1906.08172. CVPR Workshop on CVML for AR/VR.
  Describes the MediaPipe graph-execution framework underlying FaceLandmarker.

Kartynnik Y, Ablavatski A, Grishchenko I, Grundmann M (2019) Real-time facial
  surface geometry from monocular video on mobile GPUs. arXiv:1907.06724.
  Face mesh model (468 landmarks) used by MediaPipe FaceLandmarker; basis for
  the 478-landmark coordinate arrays produced by this module.
"""

import bisect
import logging
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline")

from .utils import sanitize_events_df

_MAX_REASONABLE_OFFSET_S = 10.0

_FUSION_FRONTALITY_WEIGHT_MIN = 0.0

_KEY_LANDMARK_INDICES = {    "noseTip": 1,
    "leftEye": 33,
    "rightEye": 263,
    "mouthLeft": 61,
    "mouthRight": 291,
}

_BLENDSHAPE_SIDE: Dict[str, str] = {
    "browDownLeft": "left",       "browDownRight": "right",
    "browOuterUpLeft": "left",    "browOuterUpRight": "right",
    "eyeBlinkLeft": "left",       "eyeBlinkRight": "right",
    "eyeLookDownLeft": "left",    "eyeLookDownRight": "right",
    "eyeLookInLeft": "left",      "eyeLookInRight": "right",
    "eyeLookOutLeft": "left",     "eyeLookOutRight": "right",
    "eyeLookUpLeft": "left",      "eyeLookUpRight": "right",
    "eyeSquintLeft": "left",      "eyeSquintRight": "right",
    "eyeWideLeft": "left",        "eyeWideRight": "right",
    "cheekSquintLeft": "left",    "cheekSquintRight": "right",
    "noseSneerLeft": "left",      "noseSneerRight": "right",
    "mouthSmileLeft": "left",     "mouthSmileRight": "right",
    "mouthFrownLeft": "left",     "mouthFrownRight": "right",
    "mouthDimpleLeft": "left",    "mouthDimpleRight": "right",
    "mouthStretchLeft": "left",   "mouthStretchRight": "right",
    "mouthPressLeft": "left",     "mouthPressRight": "right",
    "mouthLowerDownLeft": "left", "mouthLowerDownRight": "right",
    "mouthUpperUpLeft": "left",   "mouthUpperUpRight": "right",
    "mouthLeft": "left",          "mouthRight": "right",
}


def _estimate_frontality(landmarks_2d: np.ndarray) -> float:
    """Estimate face frontality as a [0, 1] score from normalized 2D landmarks.

    Uses the relative position of the nose tip to the eye midpoint to estimate
    yaw, and the interocular distance normalized by face height to estimate
    pitch.  A fully frontal face scores 1.0; a profile or extreme angle scores
    toward 0.0.  Returns 0.5 as fallback if landmarks array is empty.
    """
    if landmarks_2d is None or len(landmarks_2d) < 5:
        return 0.5
    required = [1, 10, 33, 152, 263]
    if any(i >= len(landmarks_2d) for i in required):
        return 0.5
    nose = landmarks_2d[1]
    left_eye = landmarks_2d[33]
    right_eye = landmarks_2d[263]
    chin = landmarks_2d[152]
    forehead = landmarks_2d[10]

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    face_height = abs(chin[1] - forehead[1]) + 1e-6
    face_width = abs(right_eye[0] - left_eye[0]) + 1e-6

    yaw_proxy = abs(nose[0] - eye_mid_x) / face_width
    pitch_proxy = abs(nose[1] - eye_mid_y) / face_height

    yaw_score = max(0.0, 1.0 - yaw_proxy * 2.0)
    pitch_score = max(0.0, 1.0 - pitch_proxy * 2.0)
    return float(yaw_score * 0.7 + pitch_score * 0.3)


def _estimate_yaw(landmarks_2d: np.ndarray) -> float:
    """Estimate signed head yaw from normalized 2D landmarks without calibration.

    Positive yaw = face turned right (participant's left side more visible to
    a frontal camera).  Negative yaw = face turned left (participant's right
    side more visible).  Uses the signed horizontal offset of the nose tip from
    the eye midpoint, normalized by interocular distance.  Range approximately
    [-1, +1] in units of interocular distance.  Returns 0.0 on failure.
    """
    if landmarks_2d is None or len(landmarks_2d) < 264:
        return 0.0
    nose_x = landmarks_2d[1][0]
    left_eye_x = landmarks_2d[33][0]
    right_eye_x = landmarks_2d[263][0]
    eye_mid_x = (left_eye_x + right_eye_x) / 2.0
    iod = abs(right_eye_x - left_eye_x) + 1e-6
    return float(np.clip((eye_mid_x - nose_x) / iod, -1.5, 1.5))


def _estimate_pitch(landmarks_2d: np.ndarray) -> float:
    """Estimate signed head pitch from normalized 2D landmarks without calibration.

    Positive pitch = face tilted downward (chin toward camera), meaning the
    camera is below or the participant is looking down, so the lower face
    region is more directly visible. Negative pitch = face tilted upward
    (forehead toward camera), so the upper face is more visible.

    Computed as the normalised vertical offset of the nose tip from the
    midpoint between forehead and chin.  In normalized image coordinates y
    increases downward.

    Returns 0.0 on failure.
    """
    if landmarks_2d is None or len(landmarks_2d) < 153:
        return 0.0
    required = [1, 10, 152]
    if any(i >= len(landmarks_2d) for i in required):
        return 0.0
    nose_y = float(landmarks_2d[1][1])
    forehead_y = float(landmarks_2d[10][1])
    chin_y = float(landmarks_2d[152][1])
    face_center_y = (forehead_y + chin_y) / 2.0
    face_height = abs(chin_y - forehead_y) + 1e-6
    return float(np.clip((nose_y - face_center_y) / face_height * 2.0, -1.5, 1.5))

_AUDIO_SAMPLE_RATE = 16000
_AUDIO_EXTRACT_DURATION_S = 30.0


@dataclass
class CameraStream:
    """Metadata and state for a single camera video file."""

    camera_index: int
    video_path: Path
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int
    time_offset_s: float = 0.0


@dataclass
class FusedFrameResult:
    """MediaPipe detection result fused across all cameras for a single logical frame."""

    frame_index: int
    timestamp_abs: float
    blendshapes: Dict[str, float]
    landmarks_2d: Optional[np.ndarray]
    landmarks_3d: Optional[np.ndarray] = None
    detection_success: bool = False
    detection_confidence: float = 0.0
    brightness: float = 0.0
    contributing_cameras: List[int] = None
    n_cameras_contributing: int = 1

    def __post_init__(self):
        """Ensure contributing_cameras is initialised to an empty list when not provided."""
        if self.contributing_cameras is None:
            self.contributing_cameras = []


class MultiCameraProcessor:
    """Process one or more camera recordings from a study prompter session."""

    def __init__(
        self,
        video_paths: List[Path],
        features_config: Dict[str, Any],
        model_path: Path,
    ):
        """Initialise the processor, open video captures, and validate that all files exist.

        Downloads the MediaPipe FaceLandmarker model if it is not already
        present at model_path. Raises FileNotFoundError if any video path does
        not exist on disk.
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.features_config = features_config
        self.model_path = Path(model_path)

        for vp in self.video_paths:
            if not vp.exists():
                raise FileNotFoundError(f"Camera video not found: {vp}")
        if not self.model_path.exists():
            from .utils import ensure_model_downloaded
            logger.info("MediaPipe model not found at %s - downloading now.", self.model_path)
            self.model_path = ensure_model_downloaded()
            logger.info("Model ready at %s", self.model_path)

        self.camera_streams: List[CameraStream] = self._open_camera_streams()
        self._landmarkers: Dict[int, Any] = {}
        self._blendshape_names: Optional[List[str]] = None

    def _open_camera_streams(self) -> List[CameraStream]:
        """Open cv2.VideoCapture for each video path, read metadata, return list."""
        streams = []
        for idx, vp in enumerate(self.video_paths):
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {vp}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            duration_sec = total_frames / fps if fps > 0 else 0.0
            streams.append(CameraStream(
                camera_index=idx,
                video_path=vp,
                fps=fps,
                total_frames=total_frames,
                duration_sec=duration_sec,
                width=width,
                height=height,
                time_offset_s=0.0,
            ))
        return streams

    def apply_offsets_from_meta(self, offsets: List[float]) -> bool:
        """Apply per-camera sync offsets read directly from recording metadata.

        Each offset is the number of seconds elapsed between the first camera
        starting and this camera starting, as recorded by the study prompter at
        capture time.  This is the preferred sync method because it uses
        ground-truth timestamps rather than audio analysis.

        Applies offsets to camera_streams in order.  Extra offsets are ignored;
        missing offsets default to 0.0.  Returns True if at least one non-zero
        offset was applied, False if all offsets are zero (no sync needed or
        single-camera session).
        """
        if not offsets:
            return False
        any_nonzero = False
        for i, stream in enumerate(self.camera_streams):
            offset = offsets[i] if i < len(offsets) else 0.0
            stream.time_offset_s = float(offset)
            if abs(offset) > 0.001:
                any_nonzero = True
                logger.info(
                    "Camera %d: applied metadata start offset %+.3f s",
                    stream.camera_index, offset,
                )
        if not any_nonzero:
            logger.info(
                "All per-camera start offsets are zero - cameras started simultaneously."
            )
        return any_nonzero

    def _extract_audio_segment(
        self,
        video_path: Path,
        duration_s: float,
        out_wav: Path,
    ) -> bool:
        """Extract the first duration_s seconds of audio from video_path to out_wav via ffmpeg.

        Returns True on success, False if ffmpeg is unavailable or fails.
        """
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-t", str(duration_s),
                    "-ac", "1",
                    "-ar", str(_AUDIO_SAMPLE_RATE),
                    "-vn",
                    str(out_wav),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
            return result.returncode == 0 and out_wav.exists()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def align_cameras(self) -> None:
        """Compute per-camera time_offset_s using audio cross-correlation.

        Extracts the first 30 seconds of audio from each camera video using
        subprocess + ffmpeg (write to temp WAV files), loads them with scipy.io.wavfile,
        resamples all to 16 kHz mono, cross-correlates each against the first
        camera to find the lag, converts lag samples to seconds, and stores the
        result in each CameraStream.time_offset_s.

        If only one camera is present, sets offset to 0.0 and returns immediately.
        If ffmpeg is unavailable or audio extraction fails for any camera, logs a
        warning, sets all offsets to 0.0, and continues without raising.
        """
        if len(self.camera_streams) == 1:
            self.camera_streams[0].time_offset_s = 0.0
            return

        already_set = any(
            abs(s.time_offset_s) > 0.001
            for s in self.camera_streams
            if s.camera_index != 0
        )
        if already_set:
            logger.info(
                "Camera offsets already applied from metadata - skipping audio sync."
            )
            return

        try:
            from scipy.io import wavfile
            from scipy.signal import correlate, resample
        except ImportError:
            logger.warning(
                "scipy not available - skipping audio sync, all camera offsets set to 0.0"
            )
            for stream in self.camera_streams:
                stream.time_offset_s = 0.0
            return

        audio_arrays: List[Optional[np.ndarray]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            wav_paths = [tmp_dir / f"cam{i}.wav" for i in range(len(self.camera_streams))]

            extraction_ok = True
            for stream, wav_path in zip(self.camera_streams, wav_paths):
                ok = self._extract_audio_segment(
                    stream.video_path, _AUDIO_EXTRACT_DURATION_S, wav_path
                )
                if not ok:
                    extraction_ok = False
                    break

            if not extraction_ok:
                logger.warning(
                    "Audio extraction failed for at least one camera - "
                    "skipping audio sync, all camera offsets set to 0.0"
                )
                for stream in self.camera_streams:
                    stream.time_offset_s = 0.0
                return

            for wav_path in wav_paths:
                try:
                    rate, data = wavfile.read(str(wav_path))
                    if data.ndim > 1:
                        data = data[:, 0]
                    data = data.astype(np.float32)
                    if rate != _AUDIO_SAMPLE_RATE:
                        n_samples = int(len(data) * _AUDIO_SAMPLE_RATE / rate)
                        data = resample(data, n_samples)
                    audio_arrays.append(data)
                except Exception as exc:
                    logger.warning(
                        "Could not load WAV %s: %s - skipping sync", wav_path, exc
                    )
                    audio_arrays.append(None)

        if any(a is None for a in audio_arrays):
            logger.warning(
                "Audio load failed for at least one camera - "
                "skipping audio sync, all camera offsets set to 0.0"
            )
            for stream in self.camera_streams:
                stream.time_offset_s = 0.0
            return

        audio_0 = audio_arrays[0]
        self.camera_streams[0].time_offset_s = 0.0

        for i, (stream, audio_i) in enumerate(
            zip(self.camera_streams[1:], audio_arrays[1:]), start=1
        ):
            min_len = min(len(audio_0), len(audio_i))
            a0 = audio_0[:min_len]
            ai = audio_i[:min_len]
            corr = correlate(a0, ai, mode="full")
            lag_samples = int(np.argmax(corr)) - (len(a0) - 1)
            time_offset_s = lag_samples / float(_AUDIO_SAMPLE_RATE)
            if abs(time_offset_s) > _MAX_REASONABLE_OFFSET_S:
                logger.warning(
                    "Camera %d: audio cross-correlation gave implausible offset %.2f s "
                    "(> %.0f s limit) - setting to 0.0. Check that camera files "
                    "are from the same recording session.",
                    i, time_offset_s, _MAX_REASONABLE_OFFSET_S,
                )
                time_offset_s = 0.0
            stream.time_offset_s = time_offset_s
            logger.info(
                "Camera %d audio offset: %+.4f s (lag %d samples)",
                i, time_offset_s, lag_samples,
            )

    def _init_landmarker(self) -> None:
        """Initialise one FaceLandmarker per camera stream (VIDEO mode).

        Each camera gets its own independent instance so that MediaPipe's
        internal temporal filter sees a consistent single-stream sequence per
        camera rather than interleaved frames from multiple views.
        """
        import mediapipe as mp

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        for stream in self.camera_streams:
            if stream.camera_index in self._landmarkers:
                try:
                    self._landmarkers[stream.camera_index].close()
                except Exception:
                    pass
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=VisionRunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.2,
                min_face_presence_confidence=0.2,
                min_tracking_confidence=0.2,
            )
            self._landmarkers[stream.camera_index] = FaceLandmarker.create_from_options(options)

    _MP_MAX_WIDTH: int = 640

    _INFERENCE_STRIDE: int = 2

    _ANN_MAX_WIDTH: int = 960

    def _run_mediapipe_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        camera_index: int = 0,
    ) -> Optional[FusedFrameResult]:
        """Run MediaPipe FaceLandmarker on a single BGR frame from the given camera.

        Each camera has its own FaceLandmarker instance so MediaPipe's VIDEO-mode
        temporal filter sees a consistent single-stream sequence.  Returns None if
        detection fails.  The frame is downscaled to at most _MP_MAX_WIDTH pixels
        wide before inference; landmark coordinates remain normalised (0–1).
        """
        import mediapipe as mp

        if camera_index not in self._landmarkers:
            self._init_landmarker()

        h, w = frame_bgr.shape[:2]
        if w > self._MP_MAX_WIDTH:
            scale = self._MP_MAX_WIDTH / w
            new_w = self._MP_MAX_WIDTH
            new_h = max(1, int(h * scale))
            inference_frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            inference_frame = frame_bgr

        rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        landmarker = self._landmarkers.get(camera_index)
        if landmarker is None:
            return None
        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            logger.debug("MediaPipe detection error at %d ms: %s", timestamp_ms, exc)
            return None

        if not result.face_blendshapes or not result.face_landmarks:
            return None

        blendshapes: Dict[str, float] = {}
        for cat in result.face_blendshapes[0]:
            blendshapes[cat.category_name] = float(cat.score)

        landmarks = result.face_landmarks[0]
        landmarks_2d = np.zeros((478, 2), dtype=np.float32)
        landmarks_3d = np.zeros((478, 3), dtype=np.float32)
        if len(landmarks) > 0:
            n = min(len(landmarks), 478)
            coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks[:n]], dtype=np.float32)
            landmarks_2d[:n, 0] = coords[:, 0]
            landmarks_2d[:n, 1] = coords[:, 1]
            landmarks_3d[:n] = coords

        brightness = float(frame_bgr[::4, ::4].mean()) / 255.0

        confidence = _estimate_frontality(landmarks_2d)

        return FusedFrameResult(
            frame_index=-1,
            timestamp_abs=-1.0,
            blendshapes=blendshapes,
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            detection_success=True,
            detection_confidence=confidence,
            brightness=brightness,
            contributing_cameras=[],
        )

    def _assign_frame_segment(
        self,
        timestamp_abs: float,
        events_df: pd.DataFrame,
    ) -> Tuple[str, int, str, int, str]:
        """Return (segment, repetition, task_group, task_id, task_name) for a timestamp.

        Uses the same interval-lookup approach as the existing capture module:
        scan events_df for the most recent 'neutral' or 'measurement' event
        whose timestamp_abs <= current timestamp, verify a 'segment_end' has
        not yet occurred for that window, otherwise assign 'inter_trial'.

        The loop walks rows in timestamp order, tracking the most recently
        seen opening event.  When a segment_end is encountered and its
        timestamp is still <= the query timestamp, it closes the current
        window.  Iteration continues to catch any later opening event.
        After the loop, if the active_event has been closed before
        timestamp_abs it becomes None.
        """
        active_event = None
        active_closed = False

        for _, row in events_df.iterrows():
            try:
                row_ts = float(row["timestamp_abs"])
            except (ValueError, TypeError):
                continue

            if row_ts > timestamp_abs:
                break

            evt = str(row["event_type"])
            if evt in ("neutral", "measurement"):
                active_event = row
                active_closed = False
            elif evt == "segment_end":
                if active_event is not None:
                    active_closed = True

        if active_event is None or active_closed:
            return ("inter_trial", 0, "0", 0, "(no task selected)")

        evt_type = str(active_event["event_type"])
        task_group = str(active_event.get("task_group", "0") or "0")
        raw_task_id = active_event.get("task_id", 0)
        try:
            task_id = int(raw_task_id)
        except (ValueError, TypeError):
            task_id = 0
        task_name = str(active_event.get("task_name", "") or "")
        raw_rep = active_event.get("repetition", 1)
        try:
            repetition = int(raw_rep) if raw_rep else 1
        except (ValueError, TypeError):
            repetition = 1

        if evt_type == "neutral":
            segment = "neutral"
        else:
            segment = "measurement"

        return (segment, repetition, task_group, task_id, task_name)

    def process_all_frames(
        self,
        events_df: pd.DataFrame,
        recording_start_offset_s: float = 0.0,
        progress_callback=None,
        video_mode: str = "none",
    ) -> Tuple[List[Dict], pd.DataFrame, List[Optional[Path]], List[Optional[Path]]]:
        """Run MediaPipe on every frame of every aligned camera, fuse results.

        Returns (frame_data_list, events_df, annotated_paths_list, landmark_paths_list).

        frame_data_list contains one dict per frame (from the primary / highest-fps
        camera after alignment) with keys matching what the existing pipeline
        expects: frame_index, timestamp_abs, segment, repetition, detection_success,
        detection_confidence, task_group, task_id, task_name, brightness, plus all
        blendshape columns (named exactly as in features.yaml) and landmark columns
        (noseTip_x/y/z, leftEye_x/y/z, rightEye_x/y/z, mouthLeft_x/y/z,
        mouthRight_x/y/z).

        Segment and task fields are filled by joining frame timestamps against
        events_df using the same logic as the existing capture module.

        Fusion strategy when multiple cameras have valid detections:
        - blendshapes: weighted mean, weights proportional to detection_confidence
        - landmarks_2d: use camera with highest detection_confidence
        - detection_success: True if any camera succeeds
        - detection_confidence: max across cameras
        """
        try:
            events_df = sanitize_events_df(events_df)
        except Exception:
            pass

        self._init_landmarker()

        primary = max(self.camera_streams, key=lambda s: s.fps)
        logger.info(
            "Primary camera for frame iteration: camera %d (%s, %.1f fps, offset %+.3f s)",
            primary.camera_index, primary.video_path.name, primary.fps, primary.time_offset_s,
        )
        primary_cap = cv2.VideoCapture(str(primary.video_path))

        n_cameras = len(self.camera_streams)
        aux_caps: List[Optional[cv2.VideoCapture]] = []
        for stream in self.camera_streams:
            if stream.camera_index == primary.camera_index:
                aux_caps.append(None)
            else:
                aux_caps.append(cv2.VideoCapture(str(stream.video_path)))

        frame_data_list: List[Dict] = []
        frame_index = 0
        _last_mp_ts_per_cam: Dict[int, int] = {s.camera_index: -1 for s in self.camera_streams}
        _aux_last_read_idx: Dict[int, int] = {}
        _no_detection_streak: int = 0
        _no_detection_streak_inference: int = 0
        _REINIT_AFTER_FRAMES: int = 90
        _last_per_cam_detection: Dict[int, Optional["FusedFrameResult"]] = {
            s.camera_index: None for s in self.camera_streams
        }
        writers: Dict[int, cv2.VideoWriter] = {}
        annotated_tmp_paths: Dict[int, Path] = {}
        landmark_writers: Dict[int, cv2.VideoWriter] = {}
        landmark_tmp_paths: Dict[int, Path] = {}
        ann_dims: Dict[int, Tuple[int, int]] = {}
        _want_annotated = video_mode in ("annotated", "both")
        _want_landmark = video_mode in ("landmark", "both")
        if _want_annotated or _want_landmark:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            for stream in self.camera_streams:
                if stream.width > self._ANN_MAX_WIDTH:
                    ann_scale = self._ANN_MAX_WIDTH / stream.width
                    ann_w = self._ANN_MAX_WIDTH
                    ann_h = max(1, int(stream.height * ann_scale))
                else:
                    ann_w, ann_h = stream.width, stream.height
                ann_dims[stream.camera_index] = (ann_w, ann_h)
                if _want_annotated:
                    try:
                        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        tmpf.close()
                        tmp_path = Path(tmpf.name)
                        w = cv2.VideoWriter(
                            str(tmp_path), fourcc, float(stream.fps), (ann_w, ann_h)
                        )
                        if w.isOpened():
                            writers[stream.camera_index] = w
                            annotated_tmp_paths[stream.camera_index] = tmp_path
                        else:
                            w.release()
                    except Exception:
                        pass
                if _want_landmark:
                    try:
                        lm_tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        lm_tmpf.close()
                        lm_path = Path(lm_tmpf.name)
                        lw = cv2.VideoWriter(
                            str(lm_path), fourcc, float(stream.fps), (ann_w, ann_h)
                        )
                        if lw.isOpened():
                            landmark_writers[stream.camera_index] = lw
                            landmark_tmp_paths[stream.camera_index] = lm_path
                        else:
                            lw.release()
                    except Exception:
                        pass

        _seg_intervals: List[list] = []
        _open_seg: Optional[list] = None
        for _, _er in events_df.sort_values("timestamp_abs", kind="stable").iterrows():
            _et = str(_er.get("event_type", ""))
            _ets = float(_er.get("timestamp_abs", 0.0))
            if _et in ("neutral", "measurement"):
                if _open_seg is not None:
                    _open_seg[1] = _ets
                try:
                    _rep = int(_er.get("repetition", 1) or 1)
                except (ValueError, TypeError):
                    _rep = 1
                try:
                    _tid = int(_er.get("task_id", 0) or 0)
                except (ValueError, TypeError):
                    _tid = 0
                _open_seg = [
                    _ets, float("inf"), _et, _rep,
                    str(_er.get("task_group", "0") or "0"),
                    _tid, str(_er.get("task_name", "") or ""),
                ]
                _seg_intervals.append(_open_seg)
            elif _et == "segment_end" and _open_seg is not None:
                _open_seg[1] = _ets
                _open_seg = None
        _seg_starts_list = [s[0] for s in _seg_intervals]

        def _draw_landmarks_on(
            frame: np.ndarray, w_px: int, h_px: int,
            landmarks_2d: Optional[np.ndarray] = None,
            dot_color: tuple = (0, 255, 0),
            label: str = "",
            conf: float = 0.0,
            seg: str = "",
        ) -> np.ndarray:
            """Draw MediaPipe landmark dots and overlay label/confidence/segment text onto a video frame."""
            ann = frame.copy()
            if landmarks_2d is not None:
                for p in landmarks_2d:
                    if p is None:
                        continue
                    px, py = float(p[0]), float(p[1])
                    if px <= 0 or py <= 0:
                        continue
                    ix = int(px * w_px)
                    iy = int(py * h_px)
                    if 0 <= ix < w_px and 0 <= iy < h_px:
                        cv2.circle(ann, (ix, iy), 1, dot_color, -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.35, w_px / 1920)
            th = max(1, int(scale * 1.5))
            y0 = int(h_px * 0.04) + 4
            line_h = int(h_px * 0.045) + 4
            def _put(txt: str, y: int, color: tuple) -> None:
                """Draw a text label with a solid black background rectangle at the given y position."""
                (tw, _), _ = cv2.getTextSize(txt, font, scale, th)
                cv2.rectangle(ann, (4, y - line_h + 2), (tw + 8, y + 4), (0, 0, 0), -1)
                cv2.putText(ann, txt, (6, y), font, scale, color, th, cv2.LINE_AA)
            if label:
                _put(label, y0, (255, 255, 255))
                y0 += line_h
            if seg and seg not in ("", "none", "0"):
                _put(f"seg:{seg}", y0, (180, 220, 255))
                y0 += line_h
            if conf > 0:
                bar_w = int((w_px - 12) * min(conf, 1.0))
                bar_y = h_px - int(h_px * 0.025) - 4
                cv2.rectangle(ann, (6, bar_y), (w_px - 6, bar_y + int(h_px * 0.018)), (40, 40, 40), -1)
                bar_color = (0, 200, 80) if conf >= 0.7 else (0, 165, 255) if conf >= 0.4 else (0, 60, 220)
                cv2.rectangle(ann, (6, bar_y), (6 + bar_w, bar_y + int(h_px * 0.018)), bar_color, -1)
            return ann

        _SKELETON_CONNECTIONS = [
            (10, 338), (338, 297), (297, 332), (332, 284),
            (284, 251), (251, 389), (389, 356), (356, 454),
            (454, 323), (323, 361), (361, 288), (288, 397),
            (397, 365), (365, 379), (379, 378), (378, 400),
            (400, 377), (377, 152), (152, 148), (148, 176),
            (176, 149), (149, 150), (150, 136), (136, 172),
            (172, 58), (58, 132), (132, 93), (93, 234),
            (234, 127), (127, 162), (162, 21), (21, 54),
            (54, 103), (103, 67), (67, 109), (109, 10),
            (33, 7), (7, 163), (163, 144), (144, 145),
            (145, 153), (153, 154), (154, 155), (155, 133),
            (33, 246), (246, 161), (161, 160), (160, 159),
            (159, 158), (158, 157), (157, 173), (173, 133),
            (263, 249), (249, 390), (390, 373), (373, 374),
            (374, 380), (380, 381), (381, 382), (382, 362),
            (263, 466), (466, 388), (388, 387), (387, 386),
            (386, 385), (385, 384), (384, 398), (398, 362),
            (61, 146), (146, 91), (91, 181), (181, 84),
            (84, 17), (17, 314), (314, 405), (405, 321),
            (321, 375), (375, 291), (61, 185), (185, 40),
            (40, 39), (39, 37), (37, 0), (0, 267),
            (267, 269), (269, 270), (270, 409), (409, 291),
            (78, 95), (95, 88), (88, 178), (178, 87),
            (87, 14), (14, 317), (317, 402), (402, 318),
            (318, 324), (324, 308), (78, 191), (191, 80),
            (80, 81), (81, 82), (82, 13), (13, 312),
            (312, 311), (311, 310), (310, 415), (415, 308),
        ]

        def _draw_skeleton_on_black(
            w_px: int, h_px: int, landmarks_2d: Optional[np.ndarray],
            label: str = "", seg: str = "",
        ) -> np.ndarray:
            """Render the facial landmark skeleton as white lines on a black background frame."""
            black = np.zeros((h_px, w_px, 3), dtype=np.uint8)
            if landmarks_2d is None:
                if label:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = max(0.35, w_px / 1920)
                    th = max(1, int(scale * 1.5))
                    y0 = int(h_px * 0.04) + 4
                    (tw, _), _ = cv2.getTextSize(label, font, scale, th)
                    cv2.rectangle(black, (4, y0 - int(h_px * 0.045)), (tw + 8, y0 + 4), (0, 0, 0), -1)
                    cv2.putText(black, label, (6, y0), font, scale, (200, 200, 200), th, cv2.LINE_AA)
                return black
            n_lm = len(landmarks_2d)
            for i, j in _SKELETON_CONNECTIONS:
                if i >= n_lm or j >= n_lm:
                    continue
                p1 = landmarks_2d[i]
                p2 = landmarks_2d[j]
                x1, y1 = int(p1[0] * w_px), int(p1[1] * h_px)
                x2, y2 = int(p2[0] * w_px), int(p2[1] * h_px)
                if (0 <= x1 < w_px and 0 <= y1 < h_px
                        and 0 <= x2 < w_px and 0 <= y2 < h_px):
                    cv2.line(black, (x1, y1), (x2, y2), (0, 200, 100), 1, cv2.LINE_AA)
            for p in landmarks_2d:
                ix, iy = int(p[0] * w_px), int(p[1] * h_px)
                if 0 <= ix < w_px and 0 <= iy < h_px:
                    cv2.circle(black, (ix, iy), 1, (255, 255, 255), -1)
            if label or seg:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = max(0.35, w_px / 1920)
                th = max(1, int(scale * 1.5))
                line_h = int(h_px * 0.045) + 4
                y0 = int(h_px * 0.04) + 4
                def _skel_put(txt: str, y: int, color: tuple) -> None:
                    """Draw a text label with a solid black background rectangle on the skeleton frame."""
                    (tw, _), _ = cv2.getTextSize(txt, font, scale, th)
                    cv2.rectangle(black, (4, y - line_h + 2), (tw + 8, y + 4), (0, 0, 0), -1)
                    cv2.putText(black, txt, (6, y), font, scale, color, th, cv2.LINE_AA)
                if label:
                    _skel_put(label, y0, (200, 200, 200))
                    y0 += line_h
                if seg and seg not in ("", "none", "0"):
                    _skel_put(f"seg:{seg}", y0, (140, 180, 220))
            return black

        while True:
            _run_inference = (frame_index % self._INFERENCE_STRIDE == 0)
            _need_decode = _run_inference or _want_annotated or _want_landmark
            if _need_decode:
                ret, primary_frame = primary_cap.read()
            else:
                ret = primary_cap.grab()
                primary_frame = None
            if not ret:
                break

            t_primary = frame_index / primary.fps
            t_wall = t_primary + primary.time_offset_s

            if _no_detection_streak_inference > 0 and _no_detection_streak_inference % _REINIT_AFTER_FRAMES == 0:
                for cam_idx in list(self._landmarkers.keys()):
                    try:
                        self._landmarkers[cam_idx].close()
                    except Exception:
                        pass
                self._landmarkers.clear()
                try:
                    self._init_landmarker()
                    _last_mp_ts_per_cam = {s.camera_index: -1 for s in self.camera_streams}
                except Exception as _reinit_err:
                    logger.warning(
                        "Failed to reinitialise landmarker after no-detection streak: %s - "
                        "face detection disabled until next reinit attempt",
                        _reinit_err,
                    )

            detections: List[Tuple[FusedFrameResult, int]] = []
            per_cam_detections: Dict[int, Optional[FusedFrameResult]] = {}

            prim_ts_ms = max(_last_mp_ts_per_cam[primary.camera_index] + 1,
                             max(1, int(frame_index / primary.fps * 1000)))

            if _run_inference:
                primary_detection = self._run_mediapipe_frame(
                    primary_frame, prim_ts_ms, camera_index=primary.camera_index)
                _last_mp_ts_per_cam[primary.camera_index] = prim_ts_ms
                _last_per_cam_detection[primary.camera_index] = primary_detection
            else:
                primary_detection = _last_per_cam_detection.get(primary.camera_index)

            per_cam_detections[primary.camera_index] = primary_detection
            if primary_detection is not None and bool(primary_detection.blendshapes):
                primary_detection.contributing_cameras = [primary.camera_index]
                detections.append((primary_detection, primary.camera_index))

            aux_frames_raw: Dict[int, np.ndarray] = {}
            for _aux_pos, (aux_stream, aux_cap) in enumerate(zip(self.camera_streams, aux_caps)):
                if aux_cap is None:
                    continue
                t_aux_local = t_wall - aux_stream.time_offset_s
                target_frame_idx = int(t_aux_local * aux_stream.fps)
                target_frame_idx = max(0, min(target_frame_idx, aux_stream.total_frames - 1))
                _expected_next = _aux_last_read_idx.get(_aux_pos, -1) + 1
                if target_frame_idx != _expected_next:
                    aux_cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_frame_idx))
                if _need_decode:
                    ret_aux, aux_frame = aux_cap.read()
                else:
                    ret_aux = aux_cap.grab()
                    aux_frame = None
                _aux_last_read_idx[_aux_pos] = target_frame_idx
                if not ret_aux:
                    per_cam_detections[aux_stream.camera_index] = None
                    continue
                if aux_frame is not None:
                    aux_frames_raw[aux_stream.camera_index] = aux_frame
                if _run_inference and aux_frame is not None:
                    aux_ts_ms = max(_last_mp_ts_per_cam.get(aux_stream.camera_index, -1) + 1,
                                    max(1, int(target_frame_idx / aux_stream.fps * 1000)))
                    d = self._run_mediapipe_frame(
                        aux_frame, aux_ts_ms, camera_index=aux_stream.camera_index)
                    _last_mp_ts_per_cam[aux_stream.camera_index] = aux_ts_ms
                    _last_per_cam_detection[aux_stream.camera_index] = d
                else:
                    d = _last_per_cam_detection.get(aux_stream.camera_index)
                per_cam_detections[aux_stream.camera_index] = d
                if d is not None and bool(d.blendshapes):
                    d.contributing_cameras = [aux_stream.camera_index]
                    detections.append((d, aux_stream.camera_index))

            if _run_inference:
                if detections:
                    _no_detection_streak_inference = 0
                else:
                    _no_detection_streak_inference += 1
            if detections:
                _no_detection_streak = 0
            else:
                _no_detection_streak += 1

            fused = self._fuse_detections(detections, frame_index, t_wall)

            _si = bisect.bisect_right(_seg_starts_list, t_wall) - 1
            if _si < 0 or _seg_intervals[_si][1] <= t_wall:
                segment, repetition, task_group, task_id, task_name = (
                    "inter_trial", 0, "0", 0, "(no task selected)"
                )
            else:
                _sv = _seg_intervals[_si]
                segment, repetition, task_group, task_id, task_name = (
                    _sv[2], _sv[3], _sv[4], _sv[5], _sv[6]
                )

            row: Dict = {
                "frame_index": frame_index,
                "timestamp_abs": t_wall,
                "segment": segment,
                "repetition": repetition,
                "detection_success": fused.detection_success,
                "detection_confidence": fused.detection_confidence,
                "task_group": task_group,
                "task_id": task_id,
                "task_name": task_name,
                "brightness": fused.brightness,
                "n_cameras_contributing": fused.n_cameras_contributing,
            }

            for bs_name, bs_val in fused.blendshapes.items():
                row[bs_name] = bs_val

            for lm_name, lm_idx in _KEY_LANDMARK_INDICES.items():
                if fused.landmarks_2d is not None and lm_idx < len(fused.landmarks_2d):
                    row[f"{lm_name}_x"] = float(fused.landmarks_2d[lm_idx, 0])
                    row[f"{lm_name}_y"] = float(fused.landmarks_2d[lm_idx, 1])
                    row[f"{lm_name}_z"] = 0.0

            if fused.landmarks_3d is not None:
                row["_landmarks_3d"] = fused.landmarks_3d.flatten().tolist()

            frame_data_list.append(row)

            fused_lm = fused.landmarks_2d if fused.detection_success else None

            try:
                if primary.camera_index in writers:
                    ann_w, ann_h = ann_dims[primary.camera_index]
                    prim_write = (cv2.resize(primary_frame, (ann_w, ann_h), interpolation=cv2.INTER_LINEAR)
                                  if (ann_w != primary.width or ann_h != primary.height)
                                  else primary_frame)
                    prim_det = per_cam_detections.get(primary.camera_index)
                    prim_lm = prim_det.landmarks_2d if prim_det is not None else None
                    _use_prim_lm = prim_lm if prim_lm is not None else fused_lm
                    if prim_lm is not None:
                        writers[primary.camera_index].write(
                            _draw_landmarks_on(prim_write, ann_w, ann_h, prim_lm, (0, 255, 0),
                                label=str(task_name or ""), conf=float(fused.detection_confidence or 0), seg=str(segment or ""))
                        )
                    elif fused_lm is not None:
                        writers[primary.camera_index].write(
                            _draw_landmarks_on(prim_write, ann_w, ann_h, fused_lm, (0, 220, 255),
                                label=str(task_name or ""), conf=float(fused.detection_confidence or 0), seg=str(segment or ""))
                        )
                    else:
                        writers[primary.camera_index].write(prim_write)
                    if primary.camera_index in landmark_writers:
                        landmark_writers[primary.camera_index].write(
                            _draw_skeleton_on_black(ann_w, ann_h, _use_prim_lm,
                                label=str(task_name or ""), seg=str(segment or ""))
                        )
                for aux_stream in self.camera_streams:
                    if aux_stream.camera_index == primary.camera_index:
                        continue
                    raw = aux_frames_raw.get(aux_stream.camera_index)
                    if raw is not None and aux_stream.camera_index in writers:
                        ann_w, ann_h = ann_dims[aux_stream.camera_index]
                        raw_write = (cv2.resize(raw, (ann_w, ann_h), interpolation=cv2.INTER_LINEAR)
                                     if (ann_w != aux_stream.width or ann_h != aux_stream.height)
                                     else raw)
                        aux_det = per_cam_detections.get(aux_stream.camera_index)
                        aux_lm = aux_det.landmarks_2d if aux_det is not None else None
                        _use_aux_lm = aux_lm if aux_lm is not None else fused_lm
                        if aux_lm is not None:
                            writers[aux_stream.camera_index].write(
                                _draw_landmarks_on(raw_write, ann_w, ann_h, aux_lm, (0, 255, 0),
                                    label=str(task_name or ""), conf=float(fused.detection_confidence or 0), seg=str(segment or ""))
                            )
                        elif fused_lm is not None:
                            writers[aux_stream.camera_index].write(
                                _draw_landmarks_on(raw_write, ann_w, ann_h, fused_lm, (0, 220, 255),
                                    label=str(task_name or ""), conf=float(fused.detection_confidence or 0), seg=str(segment or ""))
                            )
                        else:
                            writers[aux_stream.camera_index].write(raw_write)
                        if aux_stream.camera_index in landmark_writers:
                            landmark_writers[aux_stream.camera_index].write(
                                _draw_skeleton_on_black(ann_w, ann_h, _use_aux_lm,
                                    label=str(task_name or ""), seg=str(segment or ""))
                            )
            except Exception:
                pass

            frame_index += 1

            if progress_callback is not None and frame_index % 30 == 0:
                try:
                    progress_callback(frame_index, primary.total_frames)
                except Exception:
                    pass

        primary_cap.release()
        for aux_cap in aux_caps:
            if aux_cap is not None:
                aux_cap.release()

        for w in writers.values():
            try:
                w.release()
            except Exception:
                pass

        for lw in landmark_writers.values():
            try:
                lw.release()
            except Exception:
                pass

        if self._landmarkers:
            for lm in self._landmarkers.values():
                try:
                    lm.close()
                except Exception:
                    pass
            self._landmarkers.clear()

        if not frame_data_list:
            logger.warning(
                "No frames were processed. Check that the video file is readable and MediaPipe is installed correctly."
            )

        annotated_paths_list: List[Optional[Path]] = [
            annotated_tmp_paths.get(stream.camera_index)
            for stream in self.camera_streams
        ]
        landmark_paths_list: List[Optional[Path]] = [
            landmark_tmp_paths.get(stream.camera_index)
            for stream in self.camera_streams
        ]
        return frame_data_list, events_df, annotated_paths_list, landmark_paths_list

    def _fuse_detections(
        self,
        detections: List[Tuple[FusedFrameResult, int]],
        frame_index: int,
        timestamp_abs: float,
    ) -> FusedFrameResult:
        """Fuse detection results from zero or more cameras into a single result.

        When no cameras have valid detections, returns a failed detection result.
        When a single camera detects the face, uses it directly.
        When multiple cameras detect the face:

        Blendshapes: per-blendshape lateral visibility weighting using yaw.
          Left-side blendshapes weighted toward cameras whose yaw indicates the
          left face is turned toward them; right-side blendshapes analogously;
          center blendshapes equally weighted.

        Landmarks (2D and 3D): per-landmark region-aware blending.
          For each of the 478 landmarks a per-camera visibility score is
          computed from two independent components. Lateral component: how
          well this camera sees the left/right position of this specific
          landmark, using the landmark's x offset from the face midline and
          the camera's yaw estimate. Vertical component: how well this camera
          sees the upper/lower position of this specific landmark, using the
          landmark's y offset from the face vertical midpoint and the camera's
          pitch estimate. Combined visibility is the geometric mean of the two
          components. Final per-landmark weight is detection_confidence
          multiplied by combined_visibility. The computation is fully
          vectorised over all 478 landmarks via numpy, adding negligible
          overhead per frame.

          Face geometry (midline, midpoint, half-width, half-height) is derived
          from the highest-confidence camera's landmarks as the canonical
          reference frame.
        """
        if not detections:
            return FusedFrameResult(
                frame_index=frame_index,
                timestamp_abs=timestamp_abs,
                blendshapes={},
                landmarks_2d=None,
                detection_success=False,
                detection_confidence=0.0,
                brightness=0.0,
                contributing_cameras=[],
                n_cameras_contributing=0,
            )

        if len(detections) == 1:
            result, cam_idx = detections[0]
            result.frame_index = frame_index
            result.timestamp_abs = timestamp_abs
            result.n_cameras_contributing = 1
            return result

        yaws: List[float] = []
        pitches: List[float] = []
        for d, _ in detections:
            lm = d.landmarks_2d
            yaws.append(_estimate_yaw(lm) if lm is not None else 0.0)
            pitches.append(_estimate_pitch(lm) if lm is not None else 0.0)

        confidences = [d.detection_confidence for d, _ in detections]
        best_idx = int(np.argmax(confidences))
        best_detection = detections[best_idx][0]

        all_keys: set = set()
        for d, _ in detections:
            all_keys.update(d.blendshapes.keys())

        fused_blendshapes: Dict[str, float] = {}
        for key in all_keys:
            key_weights: List[float] = []
            key_vals: List[float] = []
            for i, (d, _) in enumerate(detections):
                side = _BLENDSHAPE_SIDE.get(key, "center")
                if side == "left":
                    vis = float(np.clip(0.5 + yaws[i] * 0.8, 0.05, 1.0))
                elif side == "right":
                    vis = float(np.clip(0.5 - yaws[i] * 0.8, 0.05, 1.0))
                else:
                    vis = 1.0
                key_weights.append(d.detection_confidence * vis)
                key_vals.append(d.blendshapes.get(key, 0.0))
            total_w = sum(key_weights)
            if total_w <= 0:
                fused_blendshapes[key] = sum(key_vals) / len(key_vals)
            else:
                fused_blendshapes[key] = sum(
                    v * w for v, w in zip(key_vals, key_weights)
                ) / total_w

        ref_lm = best_detection.landmarks_2d
        fused_lm2d: Optional[np.ndarray] = None
        fused_lm3d: Optional[np.ndarray] = None

        if ref_lm is not None and len(ref_lm) >= 264:
            face_cx = float((ref_lm[33, 0] + ref_lm[263, 0]) / 2.0)
            face_cy = float((ref_lm[10, 1] + ref_lm[152, 1]) / 2.0)
            face_hw = float(abs(ref_lm[263, 0] - ref_lm[33, 0])) + 1e-6
            face_hh = float(abs(ref_lm[152, 1] - ref_lm[10, 1])) / 2.0 + 1e-6

            n_lm = ref_lm.shape[0]

            left_factors = (ref_lm[:, 0] - face_cx) / face_hw
            down_factors = (ref_lm[:, 1] - face_cy) / face_hh

            weight_sum_2d = np.zeros(n_lm, dtype=np.float64)
            blended_2d = np.zeros((n_lm, 2), dtype=np.float64)

            has_3d = any(d.landmarks_3d is not None for d, _ in detections)
            weight_sum_3d = np.zeros(n_lm, dtype=np.float64)
            blended_3d = np.zeros((n_lm, 3), dtype=np.float64) if has_3d else None

            for ci, (d, _) in enumerate(detections):
                if d.landmarks_2d is None:
                    continue
                cam_lm = d.landmarks_2d
                n_this = min(cam_lm.shape[0], n_lm)

                vis_lat = np.clip(
                    0.5 + yaws[ci] * left_factors[:n_this] * 0.8, 0.05, 1.0
                )
                vis_vert = np.clip(
                    0.5 + pitches[ci] * down_factors[:n_this] * 0.8, 0.05, 1.0
                )
                vis = np.sqrt(vis_lat * vis_vert)
                w = (confidences[ci] * vis).astype(np.float64)

                weight_sum_2d[:n_this] += w
                blended_2d[:n_this] += w[:, np.newaxis] * cam_lm[:n_this].astype(np.float64)

                if blended_3d is not None and d.landmarks_3d is not None:
                    cam_lm3 = d.landmarks_3d
                    n3 = min(cam_lm3.shape[0], n_lm)
                    weight_sum_3d[:n3] += w[:n3]
                    blended_3d[:n3] += w[:n3, np.newaxis] * cam_lm3[:n3].astype(np.float64)

            fused_lm2d = np.where(
                weight_sum_2d[:, np.newaxis] > 0,
                blended_2d / np.maximum(weight_sum_2d[:, np.newaxis], 1e-12),
                ref_lm.astype(np.float64),
            ).astype(np.float32)

            if blended_3d is not None and best_detection.landmarks_3d is not None:
                fused_lm3d = np.where(
                    weight_sum_3d[:, np.newaxis] > 0,
                    blended_3d / np.maximum(weight_sum_3d[:, np.newaxis], 1e-12),
                    best_detection.landmarks_3d.astype(np.float64),
                ).astype(np.float32)
        else:
            fused_lm2d = best_detection.landmarks_2d
            fused_lm3d = best_detection.landmarks_3d

        all_mean_vis: List[float] = []
        for i, (d, _) in enumerate(detections):
            vis_vals = []
            for key in all_keys:
                side = _BLENDSHAPE_SIDE.get(key, "center")
                if side == "left":
                    vis_vals.append(float(np.clip(0.5 + yaws[i] * 0.8, 0.05, 1.0)))
                elif side == "right":
                    vis_vals.append(float(np.clip(0.5 - yaws[i] * 0.8, 0.05, 1.0)))
                else:
                    vis_vals.append(1.0)
            all_mean_vis.append(float(np.mean(vis_vals)) if vis_vals else 1.0)

        contrib_weights = [c * v for c, v in zip(confidences, all_mean_vis)]
        total_cw = sum(contrib_weights)
        if total_cw <= 0:
            fused_confidence = float(np.mean(confidences))
        else:
            fused_confidence = sum(
                c * cw for c, cw in zip(confidences, contrib_weights)
            ) / total_cw

        avg_brightness = float(np.mean([d.brightness for d, _ in detections]))
        contributing = [cam_idx for _, cam_idx in detections]

        return FusedFrameResult(
            frame_index=frame_index,
            timestamp_abs=timestamp_abs,
            blendshapes=fused_blendshapes,
            landmarks_2d=fused_lm2d,
            landmarks_3d=fused_lm3d,
            detection_success=True,
            detection_confidence=fused_confidence,
            brightness=avg_brightness,
            contributing_cameras=contributing,
            n_cameras_contributing=len(detections),
        )


def create_multi_camera_processor(
    video_paths: List[Path],
    features_config: Dict[str, Any],
    model_path: Path,
) -> MultiCameraProcessor:
    """Create and return a MultiCameraProcessor for the given camera video paths.

    This is a convenience factory that wraps the MultiCameraProcessor
    constructor. Prefer this function over direct instantiation so that
    callers are decoupled from the constructor signature.
    """
    return MultiCameraProcessor(video_paths, features_config, model_path)
