"""
Video capture module for FACET-CV facial motor and speech behaviour analysis.

Provides live camera capture and offline video file processing for study
sessions. Core responsibilities:

  - Live capture from a physical camera with face-mesh overlay rendered in
    real time using MediaPipe FaceLandmarker (IMAGE mode).
  - Keyboard-driven task and segment control (neutral baseline, measurement,
    task group/task selection, repetition counting).
  - Timestamped CaptureEvent recording and export to a pandas DataFrame that
    the pipeline's downstream stages consume.
  - Annotated video output alongside a raw camera stream, both at the true
    measured framerate so playback duration matches the recorded session.
  - Camera auto-detection that prefers a physical external camera when one is
    connected, with graceful fallback to the built-in device.

The frame_data list and events_df produced by this module are consumed
identically by the rest of the pipeline regardless of whether the data
came from live capture or offline video processing.

References
----------
Lugaresi C, Tang J, Nash H, et al. (2019) MediaPipe: A framework for building
  perception pipelines. arXiv:1906.08172. CVPR Workshop on CVML for AR/VR.
  MediaPipe graph-execution framework underlying FaceLandmarker used here.

Scott RT, Ditroilo M, Harrison AJ (2022) Concurrent validity of markerless
  motion capture for measuring the kinematics of the lower limbs during
  running gait. PeerJ 10:e13517. doi:10.7717/peerj.13517
  Validates markerless motion capture approaches as a practical alternative
  to marker-based systems in clinical and sports settings.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import platform
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

from .utils import ensure_model_downloaded, load_yaml

FACE_MESH_CONNECTIONS = frozenset([
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103),
    (103, 67), (67, 109), (109, 10), (151, 108), (108, 69), (69, 104), (104, 68), (68, 71), (71, 139),
    (139, 111), (111, 117), (117, 118), (118, 119), (119, 120), (120, 121), (121, 128), (128, 245),
    (245, 193), (193, 55), (55, 65), (65, 52), (52, 53), (53, 46), (46, 124), (124, 35), (35, 111),
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291), (291, 375), (375, 321), (321, 405),
    (405, 314), (314, 17), (17, 84), (84, 181), (181, 91), (91, 146), (146, 61), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0),
])

FACE_CONTOURS = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321),
    (321, 375), (375, 291), (291, 409), (409, 270), (270, 269), (269, 267), (267, 0), (0, 37),
    (37, 39), (39, 40), (40, 185), (185, 61),
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133),
    (133, 155), (155, 154), (154, 153), (153, 145), (145, 144), (144, 163), (163, 7), (7, 33),
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249), (249, 263),
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107),
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336),
])


@dataclass
class CaptureEvent:
    """A timestamped event recorded during video capture."""
    event_type: str
    timestamp_abs: float
    frame_index: int
    label: str
    task_group: Optional[str] = None
    task_id: Optional[int] = None
    task_name: Optional[str] = None


@dataclass
class CaptureConfig:
    """Hardware and layout parameters for video capture."""
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: float = 60.0
    codec: str = "mp4v"
    panel_width: int = 300


class VideoCapture:
    """Manages live or file-based video capture with face-mesh overlay and event annotation."""

    def __init__(self, config: CaptureConfig, plotting_config: Dict[str, Any]):
        """Initialise the VideoCapture instance.

        Downloads the MediaPipe FaceLandmarker model if needed, initialises
        the landmarker in IMAGE mode for single-frame inference, and sets up
        internal state for events, frame data, and task tracking.
        """
        self.config = config
        self.plotting_config = plotting_config

        model_path = ensure_model_downloaded()
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self.events: List[CaptureEvent] = []
        self.frame_data: List[Dict] = []
        self.current_segment: Optional[str] = None
        self.segment_start_frame: int = 0
        self.repetition_count: int = 0
        self.task_rep_counts: Dict[Tuple[Optional[str], Optional[int]], int] = {}
        self.start_time: float = 0.0

        self.current_task_group: Optional[str] = None
        self.current_task_id: Optional[int] = None
        self.current_task_name: str = "(no task selected)"
        self.tasks_config: Dict = self._load_tasks_config()

        self.live_metrics: Dict[str, float] = {
            "fps": 0.0,
            "detection_confidence": 0.0,
            "current_segment": "none",
            "frame_count": 0,
            "current_task": "(no task selected)",
        }

        self.recent_event: Optional[Tuple[str, float]] = None
        self._focus_warned: bool = False

    def _load_tasks_config(self) -> Dict:
        """Load tasks configuration from YAML file."""
        config_path = Path(__file__).parent.parent / "config" / "tasks.yaml"
        if config_path.exists():
            return load_yaml(config_path)
        return {}

    def _select_task_group(self, group: str, timestamp: float, frame_index: int) -> None:
        """Select a task group (A, B, or C) and reset task state."""
        if group not in self.tasks_config.get("task_groups", {}):
            return
        self.current_task_group = group
        self.current_task_id = None
        group_name = self.tasks_config["task_groups"][group].get("name", group)
        self.current_task_name = f"Group {group}: {group_name}"
        self.live_metrics["current_task"] = self.current_task_name
        self.repetition_count = self.task_rep_counts.get(
            (self.current_task_group, self.current_task_id), 0)
        print(f"[{timestamp:.2f}s] Selected: {self.current_task_name}")
        self.recent_event = (f"GROUP {group} SELECTED", timestamp)

    def _select_task(self, task_num: int, timestamp: float, frame_index: int) -> None:
        """Select a specific task within the current group."""
        if not self.current_task_group:
            print(f"[{timestamp:.2f}s] Select a task group first (A, B, or C)")
            return

        group_tasks = (
            self.tasks_config
            .get("task_groups", {})
            .get(self.current_task_group, {})
            .get("tasks", {})
        )
        if task_num not in group_tasks:
            print(f"[{timestamp:.2f}s] Task {task_num} not found in group {self.current_task_group}")
            return

        self.current_task_id = task_num
        task_info = group_tasks[task_num]
        display_name = task_info.get("display_name", task_info.get("name", f"Task {task_num}"))
        self.current_task_name = f"{self.current_task_group}{task_num}: {display_name}"
        self.live_metrics["current_task"] = self.current_task_name
        self.repetition_count = self.task_rep_counts.get(
            (self.current_task_group, self.current_task_id), 0)
        print(f"[{timestamp:.2f}s] Selected: {self.current_task_name}")
        self.recent_event = (f"TASK: {display_name}", timestamp)

    def _parse_and_set_task(self, task_info: str) -> None:
        """Parse a task identifier string (e.g. 'A2') and set task state."""
        task_info = task_info.strip().upper()
        if len(task_info) < 2:
            print(f"Invalid task identifier: {task_info}")
            return

        group = task_info[0]
        try:
            task_num = int(task_info[1:])
        except ValueError:
            print(f"Invalid task number in: {task_info}")
            return

        if group not in self.tasks_config.get("task_groups", {}):
            print(f"Unknown task group: {group}")
            return

        group_tasks = (
            self.tasks_config.get("task_groups", {}).get(group, {}).get("tasks", {})
        )
        if task_num not in group_tasks:
            print(f"Task {task_num} not found in group {group}")
            return

        self.current_task_group = group
        self.current_task_id = task_num
        task_data = group_tasks[task_num]
        display_name = task_data.get("display_name", task_data.get("name", f"Task {task_num}"))
        self.current_task_name = f"{group}{task_num}: {display_name}"
        self.live_metrics["current_task"] = self.current_task_name
        print(f"Task set: {self.current_task_name}")

    @staticmethod
    def _get_camera_backend() -> int:
        """Return the preferred OpenCV backend for the current platform."""
        if platform.system() == "Darwin":
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_ANY

    @staticmethod
    def _list_physical_cameras_macos() -> List[Tuple[int, str]]:
        """Parse system_profiler on macOS to identify physical camera names.

        Returns a list of (probe_order, name) tuples for physical cameras.
        Virtual cameras (names containing 'virtual' or 'obs') are excluded.
        The indices are enumeration order WITHIN physical cameras only and are
        NOT used as OpenCV indices - actual indices are determined by probing.
        """
        try:
            result = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return []
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        virtual_keywords = ("virtual", "obs")
        cameras: List[Tuple[int, str]] = []
        phys_idx = 0
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.endswith(":") and stripped != "Camera:":
                name = stripped.rstrip(":")
                is_virtual = any(kw in name.lower() for kw in virtual_keywords)
                if not is_virtual:
                    cameras.append((phys_idx, name))
                    phys_idx += 1
        return cameras

    @staticmethod
    def _is_real_camera(cap: cv2.VideoCapture, warmup: int = 5) -> bool:
        """Read several frames and return True if inter-frame variance is non-zero.

        A virtual camera displaying a static logo produces identical frames
        (variance ~ 0), while a real sensor always has some photon / readout
        noise between consecutive captures.
        """
        frames = []
        for _ in range(warmup):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            time.sleep(0.05)

        if len(frames) < 2:
            return False

        diffs = [
            np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)))
            for i in range(1, len(frames))
        ]
        return float(np.mean(diffs)) > 0.5

    def _detect_best_camera(self) -> Tuple[int, int]:
        """Probe available cameras and return (camera_index, backend).

        On macOS, system_profiler is consulted first to determine the total
        number of cameras and whether a physical external camera exists.
        The probe range is limited to the actual device count (instead of a
        hard-coded 10) to avoid "index out of bound" errors that can
        destabilise USB cameras.  Each camera is opened only ONCE during
        probing: the real-camera check and resolution read happen in the same
        session to minimise open/close cycles on USB devices.
        """
        backend = self._get_camera_backend()

        prefer_external = False
        external_label = "external"
        max_probe = 10

        if platform.system() == "Darwin":
            physical = self._list_physical_cameras_macos()
            try:
                result = subprocess.run(
                    ["system_profiler", "SPCameraDataType"],
                    capture_output=True, text=True, timeout=5,
                )
                total_devices = sum(
                    1 for line in result.stdout.splitlines()
                    if line.strip().endswith(":") and line.strip() != "Camera:"
                )
                max_probe = max(total_devices, 1)
            except Exception:
                max_probe = max(len(physical), 4)

            builtin_kw = ("facetime", "isight")
            external_names = [
                name for _, name in physical
                if not any(kw in name.lower() for kw in builtin_kw)
            ]
            if external_names:
                prefer_external = True
                external_label = external_names[0]

        candidates: List[Tuple[int, float, float]] = []
        for idx in range(max_probe):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if not self._is_real_camera(cap):
                cap.release()
                continue
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            candidates.append((idx, float(w), float(h)))

        if not candidates:
            print("No working cameras found, falling back to default index")
            return (self.config.camera_id, backend)

        if len(candidates) == 1:
            best = candidates[0]
            print(f"Selected camera: index {best[0]}, {int(best[1])}x{int(best[2])}")
            return (best[0], backend)

        if prefer_external:
            non_zero = [c for c in candidates if c[0] > 0]
            if non_zero:
                best = max(non_zero, key=lambda c: c[1] * c[2])
                print(
                    f"Selected external camera: {external_label} "
                    f"(index {best[0]}, {int(best[1])}x{int(best[2])})"
                )
                return (best[0], backend)

        best = max(candidates, key=lambda c: (c[1] * c[2], c[0]))
        label = "external" if best[0] > 0 else "built-in"
        print(
            f"Selected camera: index {best[0]} ({label}), "
            f"{int(best[1])}x{int(best[2])}"
        )
        return (best[0], backend)

    def capture_live(
        self,
        output_video_path: Path,
        output_annotated_path: Path,
        output_normal_video_path: Path,
        output_normal_annotated_path: Path,
    ) -> Tuple[List[Dict], List[CaptureEvent]]:
        """Capture live video from camera, recording raw and annotated streams.

        After capture completes, normal-speed copies of both streams are created
        at the true measured framerate so playback matches real-time duration.
        """
        camera_idx, backend = self._detect_best_camera()

        if camera_idx > 0:
            time.sleep(0.8)

        cap = cv2.VideoCapture(camera_idx, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera (index={camera_idx}, "
                f"backend={'AVFoundation' if backend == cv2.CAP_AVFOUNDATION else 'auto'}). "
                f"Check System Settings > Privacy & Security > Camera."
            )

        for _ in range(5):
            ret, _ = cap.read()
            if ret:
                break
            time.sleep(0.1)

        if not ret:
            cap.release()
            raise RuntimeError(
                f"Camera opened but produced no frames (index={camera_idx}). "
                f"Ensure the camera is not in use by another application."
            )

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or self.config.fps

        camera_label = "external" if camera_idx > 0 else "built-in"
        print(
            f"Camera ready: index {camera_idx} ({camera_label}), "
            f"{actual_width}x{actual_height} @ {actual_fps:.0f} fps"
        )

        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        raw_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, actual_fps, (actual_width, actual_height)
        )
        annotated_width = actual_width + self.config.panel_width
        annotated_writer = cv2.VideoWriter(
            str(output_annotated_path), fourcc, actual_fps, (annotated_width, actual_height)
        )

        self.start_time = time.time()
        frame_index = 0
        prev_time = self.start_time

        print("\n=== Live Capture Started ===")
        print("Controls:")
        print("  'a/b/c' - Select task group (A/B/C)")
        print("  '1-9'   - Select task within group")
        print("  'n'     - Mark neutral baseline segment")
        print("  'm'     - Mark measurement segment")
        print("  'r'     - End current segment / next repetition")
        print("  'f'     - Toggle fullscreen")
        print("  'q'/ESC - Stop capture")
        print("===================================\n")

        window_name = "Facial Motor Assessment - Live Capture"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, annotated_width, actual_height)

        is_fullscreen = False
        screen_width, screen_height = None, None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()
                elapsed = current_time - self.start_time
                dt = current_time - prev_time
                self.live_metrics["fps"] = 1.0 / dt if dt > 0 else 0.0
                prev_time = current_time

                raw_writer.write(frame)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.face_landmarker.detect(mp_image)

                detected = len(results.face_landmarks) > 0
                confidence = 1.0 if detected else 0.0

                frame_info = {
                    "frame_index": frame_index,
                    "timestamp_abs": elapsed,
                    "detection_success": detected,
                    "detection_confidence": confidence,
                    "segment": self.current_segment,
                    "repetition": self.repetition_count if self.current_segment == "measurement" else 0,
                    "task_group": self.current_task_group,
                    "task_id": self.current_task_id,
                    "task_name": self.current_task_name,
                }

                self.frame_data.append(frame_info)
                self.live_metrics["detection_confidence"] = confidence
                self.live_metrics["frame_count"] = frame_index
                self.live_metrics["current_segment"] = self.current_segment or "none"

                annotated_frame = self._draw_annotations(frame.copy(), results, elapsed)
                display_frame = self._create_display_frame(annotated_frame, actual_height)
                annotated_writer.write(display_frame)

                if is_fullscreen and screen_width and screen_height:
                    display_frame = self._scale_to_fullscreen(
                        display_frame, screen_width, screen_height
                    )

                cv2.imshow(window_name, display_frame)

                key_raw = cv2.waitKey(1)
                if key_raw != -1:
                    self._handle_keypress(
                        key_raw, elapsed, frame_index, window_name,
                        annotated_width, actual_height,
                        is_fullscreen, screen_width, screen_height,
                    )
                    key = key_raw & 0xFF
                    if chr(key).lower() == "q" or key_raw == 27:
                        if self.current_segment:
                            self._handle_event("segment_end", elapsed, frame_index)
                        break
                    if chr(key).lower() == "f":
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            cv2.setWindowProperty(
                                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                            )
                            cv2.waitKey(50)
                            screen_width, screen_height = self._detect_screen_size()
                            print(f"Fullscreen mode: {screen_width}x{screen_height}")
                        else:
                            cv2.setWindowProperty(
                                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                            )
                            cv2.resizeWindow(window_name, annotated_width, actual_height)
                            print("Windowed mode")
                    self._focus_warned = False
                else:
                    if not self._focus_warned and frame_index > 100 and frame_index % 300 == 0:
                        print("Hint: Click the video window to ensure keyboard shortcuts (n/m/r) are captured.")
                        self._focus_warned = True

                frame_index += 1

        finally:
            elapsed_total = time.time() - self.start_time
            true_fps = frame_index / elapsed_total if elapsed_total > 0 else self.config.fps

            cap.release()
            raw_writer.release()
            annotated_writer.release()
            cv2.destroyAllWindows()

            print(f"\nLive session complete.")
            print(f"Recorded FPS: {actual_fps:.1f}, True FPS: {true_fps:.1f}")
            print(f"Raw video saved to: {output_video_path.resolve()}")
            print(f"Annotated video saved to: {output_annotated_path.resolve()}")

            self._create_normal_speed_copy(
                output_video_path, output_normal_video_path, true_fps,
            )
            print(f"Normal-speed raw video saved to: {output_normal_video_path.resolve()}")

            self._create_normal_speed_copy(
                output_annotated_path, output_normal_annotated_path, true_fps,
            )
            print(f"Normal-speed annotated video saved to: {output_normal_annotated_path.resolve()}")

        return self.frame_data, self.events

    @staticmethod
    def _create_normal_speed_copy(
        source_path: Path, dest_path: Path, true_fps: float,
    ) -> None:
        """Re-encode a video file at the given framerate to produce a normal-speed copy."""
        cap = cv2.VideoCapture(str(source_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dest_path), fourcc, true_fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        cap.release()
        writer.release()

    def process_video_file(
        self,
        input_path: Path,
        output_annotated_path: Path,
        output_normal_video_path: Path,
        task_info: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Tuple[List[Dict], List[CaptureEvent]]:
        """Process a pre-recorded video file with optional task and time-range constraints.

        A clean (non-annotated) copy of the processed frame range is always saved
        alongside the annotated output at the original framerate.
        """
        if task_info:
            self._parse_and_set_task(task_info)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps

        start_frame = max(0, min(int(start_time * fps) if start_time else 0, total_frames - 1))
        end_frame = max(start_frame + 1, min(int(end_time * fps) if end_time else total_frames, total_frames))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        writer = cv2.VideoWriter(str(output_annotated_path), fourcc, fps, (width, height))

        normal_writer = cv2.VideoWriter(
            str(output_normal_video_path), fourcc, fps, (width, height),
        )

        frame_index = 0
        video_frame_index = start_frame
        frames_to_process = end_frame - start_frame

        print(f"\nProcessing video: {input_path.name}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")
        if start_time or end_time:
            print(f"Time range: {start_time or 0:.1f}s to {end_time or total_duration:.1f}s "
                  f"(frames {start_frame}-{end_frame})")

        try:
            while video_frame_index < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_index / fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.face_landmarker.detect(mp_image)

                detected = len(results.face_landmarks) > 0
                self.frame_data.append({
                    "frame_index": frame_index,
                    "timestamp_abs": timestamp,
                    "detection_success": detected,
                    "detection_confidence": 1.0 if detected else 0.0,
                    "segment": "measurement",
                    "repetition": 0,
                    "task_group": self.current_task_group,
                    "task_id": self.current_task_id,
                    "task_name": self.current_task_name,
                })

                normal_writer.write(frame)

                annotated_frame = self._draw_annotations(frame.copy(), results, timestamp)
                writer.write(annotated_frame)

                if frame_index % 100 == 0:
                    progress = (frame_index / frames_to_process) * 100
                    print(f"Progress: {progress:.1f}% ({frame_index}/{frames_to_process})")

                frame_index += 1
                video_frame_index += 1

        finally:
            cap.release()
            writer.release()
            normal_writer.release()

        print(f"Video processing complete. {frame_index} frames processed.")
        print(f"Normal-speed video saved to: {output_normal_video_path.resolve()}")
        return self.frame_data, self.events

    def _handle_event(self, event_type: str, timestamp: float, frame_index: int) -> None:
        """Record a capture event and update internal segment state.

        Supported event_type values are 'neutral', 'measurement', and
        'segment_end'. Transitions an open segment to segment_end automatically
        before opening a new neutral or measurement window. Increments the
        per-task repetition counter on each new measurement event.
        """
        if event_type == "neutral":
            if self.current_segment == "neutral":
                return
            if self.current_segment:
                self.events.append(
                    CaptureEvent("segment_end", timestamp, frame_index, f"end_{self.current_segment}")
                )
            self.current_segment = "neutral"
            self.segment_start_frame = frame_index
            label = "NEUTRAL BASELINE"

        elif event_type == "measurement":
            if self.current_segment == "measurement":
                return
            if self.current_segment:
                self.events.append(
                    CaptureEvent("segment_end", timestamp, frame_index, f"end_{self.current_segment}")
                )
            self.current_segment = "measurement"
            self.segment_start_frame = frame_index
            task_key = (self.current_task_group, self.current_task_id)
            self.task_rep_counts[task_key] = self.task_rep_counts.get(task_key, 0) + 1
            self.repetition_count = self.task_rep_counts[task_key]
            if self.current_task_id:
                label = f"MEASUREMENT: {self.current_task_name} (Rep {self.repetition_count})"
            else:
                label = f"MEASUREMENT (Rep {self.repetition_count})"

        elif event_type == "segment_end":
            if not self.current_segment:
                return
            label = f"END {self.current_segment.upper()}"
            self.events.append(CaptureEvent("segment_end", timestamp, frame_index, label))
            self.current_segment = None
            return
        else:
            return

        self.events.append(CaptureEvent(
            event_type, timestamp, frame_index, label,
            self.current_task_group, self.current_task_id, self.current_task_name,
        ))
        self.recent_event = (label, timestamp)
        print(f"[{timestamp:.2f}s] Event: {label}")

    def _handle_keypress(
        self, key_raw: int, elapsed: float, frame_index: int,
        window_name: str, annotated_width: int, actual_height: int,
        is_fullscreen: bool, screen_width: Optional[int], screen_height: Optional[int],
    ) -> None:
        """Dispatch keyboard input to the appropriate handler."""
        key = key_raw & 0xFF
        try:
            ch = chr(key).lower()
        except Exception:
            return

        if ch == "n":
            self._handle_event("neutral", elapsed, frame_index)
        elif ch == "m":
            self._handle_event("measurement", elapsed, frame_index)
        elif ch == "r":
            self._handle_event("segment_end", elapsed, frame_index)
        elif ch == "a":
            self._select_task_group("A", elapsed, frame_index)
        elif ch == "b":
            self._select_task_group("B", elapsed, frame_index)
        elif ch == "c":
            self._select_task_group("C", elapsed, frame_index)
        elif ch.isdigit() and "1" <= ch <= "9":
            self._select_task(int(ch), elapsed, frame_index)

    def _draw_annotations(self, frame: np.ndarray, results, timestamp: float) -> np.ndarray:
        """Draw face-mesh connections, contours, and status text onto the frame."""
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                h, w = frame.shape[:2]
                for connection in FACE_MESH_CONNECTIONS:
                    s_idx, e_idx = connection
                    if s_idx < len(face_landmarks) and e_idx < len(face_landmarks):
                        s_pt = (int(face_landmarks[s_idx].x * w), int(face_landmarks[s_idx].y * h))
                        e_pt = (int(face_landmarks[e_idx].x * w), int(face_landmarks[e_idx].y * h))
                        cv2.line(frame, s_pt, e_pt, (192, 192, 192), 1)

                for connection in FACE_CONTOURS:
                    s_idx, e_idx = connection
                    if s_idx < len(face_landmarks) and e_idx < len(face_landmarks):
                        s_pt = (int(face_landmarks[s_idx].x * w), int(face_landmarks[s_idx].y * h))
                        e_pt = (int(face_landmarks[e_idx].x * w), int(face_landmarks[e_idx].y * h))
                        cv2.line(frame, s_pt, e_pt, (0, 255, 0), 1)

        if self.recent_event:
            event_label, event_time = self.recent_event
            if timestamp - event_time < 2.0:
                cv2.putText(frame, event_label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                self.recent_event = None

        cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.current_segment:
            color = (0, 255, 0) if self.current_segment == "measurement" else (255, 255, 0)
            cv2.putText(
                frame, f"Recording: {self.current_segment.upper()}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        return frame

    def _create_display_frame(self, frame: np.ndarray, height: int) -> np.ndarray:
        """Create a side-panel display frame with live metrics and controls."""
        panel = np.full((height, self.config.panel_width, 3), 30, dtype=np.uint8)
        y = 40
        lh = 35

        cv2.putText(panel, "LIVE METRICS", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += lh + 10

        cv2.putText(panel, f"FPS: {self.live_metrics['fps']:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += lh

        conf = self.live_metrics["detection_confidence"]
        conf_color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
        cv2.putText(panel, f"Detection: {conf:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 1)
        y += lh

        cv2.putText(panel, f"Frames: {self.live_metrics['frame_count']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += lh

        segment = self.live_metrics["current_segment"]
        seg_color = (0, 255, 0) if segment == "measurement" else (255, 255, 0) if segment == "neutral" else (128, 128, 128)
        cv2.putText(panel, f"Segment: {segment}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, seg_color, 1)
        y += lh

        active_rep = self.repetition_count if self.current_segment == "measurement" else 0
        cv2.putText(panel, f"Repetition: {active_rep}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += lh

        task_str = self.live_metrics.get("current_task", "(no task selected)")
        if len(task_str) > 25:
            task_str = task_str[:22] + "..."
        task_color = (0, 200, 255) if self.current_task_id else (128, 128, 128)
        cv2.putText(panel, f"Task: {task_str}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, task_color, 1)
        y += lh + 10

        cv2.line(panel, (10, y), (self.config.panel_width - 10, y), (100, 100, 100), 1)
        y += 20

        cv2.putText(panel, "CONTROLS", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += lh

        for ctrl in (
            "'a/b/c' - Task group",
            "'1-9' - Select task",
            "'n' - Neutral baseline",
            "'m' - Measurement",
            "'r' - End segment",
            "'q' - Stop capture",
        ):
            cv2.putText(panel, ctrl, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y += 25

        return np.hstack([frame, panel])

    @staticmethod
    def _scale_to_fullscreen(
        frame: np.ndarray, screen_width: int, screen_height: int
    ) -> np.ndarray:
        """Scale a frame to fill the screen while preserving aspect ratio."""
        canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        fh, fw = frame.shape[:2]
        scale = min(screen_width / fw, screen_height / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x_off = (screen_width - new_w) // 2
        y_off = (screen_height - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = scaled
        return canvas

    @staticmethod
    def _detect_screen_size() -> Tuple[int, int]:
        """Detect screen resolution on macOS, falling back to 1920x1080."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=2,
            )
            for line in result.stdout.split("\n"):
                if "Resolution" in line and "Retina" not in line:
                    parts = line.split(":")[1].strip().split(" x ")
                    if len(parts) >= 2:
                        return int(parts[0].strip()), int(parts[1].split()[0].strip())
        except Exception:
            pass
        return 1920, 1080

    def get_events_dataframe(self) -> pd.DataFrame:
        """Return all captured events as a pandas DataFrame.

        Returns an empty DataFrame with the correct column schema when no
        events have been recorded. Columns are: event_type, timestamp_abs,
        frame_index, label, task_group, task_id, task_name.
        """
        if not self.events:
            return pd.DataFrame(columns=[
                "event_type", "timestamp_abs", "frame_index", "label",
                "task_group", "task_id", "task_name",
            ])
        return pd.DataFrame([
            {
                "event_type": e.event_type,
                "timestamp_abs": e.timestamp_abs,
                "frame_index": e.frame_index,
                "label": e.label,
                "task_group": e.task_group,
                "task_id": e.task_id,
                "task_name": e.task_name,
            }
            for e in self.events
        ])


def create_capture(plotting_config: Dict[str, Any], camera_id: int = 0) -> VideoCapture:
    """Create and return a VideoCapture instance using the given plotting config.

    Reads panel_width from plotting_config['live_display'] if present, otherwise
    uses the CaptureConfig default of 300 pixels. This factory is the preferred
    way to construct a VideoCapture so that callers do not need to build a
    CaptureConfig directly.
    """
    live_config = plotting_config.get("live_display", {})
    config = CaptureConfig(
        camera_id=camera_id,
        panel_width=live_config.get("panel_width", 300),
    )
    return VideoCapture(config, plotting_config)
