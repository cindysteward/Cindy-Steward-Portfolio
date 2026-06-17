"""
Video file processor for offline analysis of pre-recorded sessions in FACET-CV.

Handles video files using one of three annotation strategies:

  1. Manual task annotations loaded from a JSON file.
  2. Automatic motion-based segment detection using Farneback optical flow.
  3. Continuous processing (no task structure; first 5 s treated as neutral).

All three strategies produce frame_data and events_df in the same format as
live capture (capture.py), so the rest of the pipeline works identically
regardless of the data source.

Deprecation note
----------------
The VideoFileProcessor class (motion-based and manual JSON annotation) remains
for backwards compatibility with the --video, --annotations, --auto-detect, and
--continuous CLI flags. For study-prompter multi-camera recordings, use
multi_camera_processor.MultiCameraProcessor together with study_prompter_reader
instead of this class.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .utils import load_json, save_json


@dataclass
class TaskAnnotation:
    """Single task execution with repetitions."""
    task_group: str
    task_id: int
    task_name: str
    repetitions: List[Dict[str, float]]


@dataclass
class VideoAnnotation:
    """Complete annotation for a video file."""
    video_path: str
    fps: float
    total_frames: int
    duration_sec: float
    neutral_segments: List[Dict[str, float]]
    tasks: List[TaskAnnotation]
    is_continuous: bool = False


class VideoFileProcessor:
    """Process pre-recorded video files for FACET-CV facial motor analysis.

    Reads video metadata on construction and exposes methods for loading
    annotations, detecting segments automatically from motion, generating
    continuous-mode annotations, and producing the frame_data and events_df
    structures consumed by the rest of the pipeline.
    """

    def __init__(self, video_path: Path):
        """Initialise the processor and read video metadata from the file.

        Raises FileNotFoundError if the video file does not exist. Stores fps,
        total_frames, duration_sec, width, and height for use by downstream
        methods.
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"Video loaded: {self.video_path.name}")
        print(f"  Duration: {self.duration_sec:.1f}s, Frames: {self.total_frames}, FPS: {self.fps:.1f}")

    def load_annotations(self, annotation_path: Path) -> VideoAnnotation:
        """Load manual task annotations from a JSON file.

        Expected JSON format::

            {
              "is_continuous": false,
              "neutral_segments": [
                {"start_time": 0.0, "end_time": 5.0}
              ],
              "tasks": [
                {
                  "task_group": "A",
                  "task_id": 1,
                  "task_name": "Smiling Broadly",
                  "repetitions": [
                    {"start_time": 6.0, "end_time": 9.0},
                    {"start_time": 10.0, "end_time": 13.0}
                  ]
                }
              ]
            }
        """
        data = load_json(annotation_path)
        is_continuous = data.get('is_continuous', False)

        tasks = []
        for task_data in data.get('tasks', []):
            tasks.append(TaskAnnotation(
                task_group=task_data['task_group'],
                task_id=task_data['task_id'],
                task_name=task_data['task_name'],
                repetitions=task_data['repetitions'],
            ))

        return VideoAnnotation(
            video_path=str(self.video_path),
            fps=self.fps,
            total_frames=self.total_frames,
            duration_sec=self.duration_sec,
            neutral_segments=data.get('neutral_segments', []),
            tasks=tasks,
            is_continuous=is_continuous,
        )

    def auto_detect_segments(
        self,
        motion_threshold: float = 0.10,
        min_segment_duration: float = 1.5,
        min_rest_duration: float = 1.0,
    ) -> VideoAnnotation:
        """Automatically detect task segments using optical-flow motion analysis.

        Returns a VideoAnnotation with auto-detected segments marked for manual
        labelling.
        """
        print("Auto-detecting segments from motion...")

        cap = cv2.VideoCapture(str(self.video_path))
        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read video")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_scores = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_scores.append(np.mean(magnitude))

            prev_gray = gray
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  Analyzed {frame_idx}/{self.total_frames} frames...")

        cap.release()

        motion_scores = np.array(motion_scores)
        if motion_scores.max() > 0:
            motion_scores = motion_scores / motion_scores.max()

        is_active = motion_scores > motion_threshold
        segments: List[Dict[str, float]] = []
        in_segment = False
        segment_start = 0

        for i, active in enumerate(is_active):
            if active and not in_segment:
                segment_start = i
                in_segment = True
            elif not active and in_segment:
                segment_end = i
                duration = (segment_end - segment_start) / self.fps
                if duration >= min_segment_duration:
                    segments.append({
                        'start_time': segment_start / self.fps,
                        'end_time': segment_end / self.fps,
                    })
                in_segment = False

        if in_segment:
            duration = (len(is_active) - segment_start) / self.fps
            if duration >= min_segment_duration:
                segments.append({
                    'start_time': segment_start / self.fps,
                    'end_time': len(is_active) / self.fps,
                })

        print(f"Detected {len(segments)} motion segments")

        neutral_segments: List[Dict[str, float]] = []
        if segments and segments[0]['start_time'] > min_rest_duration:
            neutral_segments.append({
                'start_time': 0.0,
                'end_time': segments[0]['start_time'],
            })

        tasks = [TaskAnnotation(
            task_group='AUTO',
            task_id=0,
            task_name='Auto-detected (LABEL THIS)',
            repetitions=segments,
        )]

        return VideoAnnotation(
            video_path=str(self.video_path),
            fps=self.fps,
            total_frames=self.total_frames,
            duration_sec=self.duration_sec,
            neutral_segments=neutral_segments,
            tasks=tasks,
            is_continuous=False,
        )

    def create_continuous_annotation(self) -> VideoAnnotation:
        """Create annotation for continuous (non-task-based) video.

        Treats the first 5 seconds (or 10 % of duration if shorter) as neutral
        baseline and the remainder as measurement.
        """
        neutral_duration = min(5.0, self.duration_sec * 0.1)

        return VideoAnnotation(
            video_path=str(self.video_path),
            fps=self.fps,
            total_frames=self.total_frames,
            duration_sec=self.duration_sec,
            neutral_segments=[{'start_time': 0.0, 'end_time': neutral_duration}],
            tasks=[],
            is_continuous=True,
        )

    def generate_frame_data(self, annotation: VideoAnnotation) -> List[Dict]:
        """Generate a frame_data list in the same format produced by capture.py.

        Iterates over every frame in the video and assigns each one a segment
        ('neutral', 'measurement', or 'inter_trial'), repetition index, and
        task fields based on the provided annotation. Returns one dict per
        frame with keys: frame_index, timestamp_abs, segment, repetition,
        task_group, task_id, task_name.
        """
        frame_data: List[Dict] = []

        for frame_idx in range(self.total_frames):
            timestamp = frame_idx / self.fps

            segment = None
            repetition = 0
            task_group = None
            task_id = None
            task_name = None

            for neutral_seg in annotation.neutral_segments:
                if neutral_seg['start_time'] <= timestamp <= neutral_seg['end_time']:
                    segment = 'neutral'
                    break

            if segment is None and not annotation.is_continuous:
                for task in annotation.tasks:
                    for rep_idx, rep in enumerate(task.repetitions, start=1):
                        if rep['start_time'] <= timestamp <= rep['end_time']:
                            segment = 'measurement'
                            repetition = rep_idx
                            task_group = task.task_group
                            task_id = task.task_id
                            task_name = task.task_name
                            break
                    if segment is not None:
                        break

            if annotation.is_continuous and segment is None:
                segment = 'measurement'
                repetition = 1
                task_group = '0'
                task_id = 0
                task_name = 'continuous'

            if segment is None:
                segment = 'inter_trial'

            frame_data.append({
                'frame_index': frame_idx,
                'timestamp_abs': timestamp,
                'segment': segment,
                'repetition': repetition,
                'task_group': task_group,
                'task_id': task_id,
                'task_name': task_name,
            })

        return frame_data

    def generate_events_df(self, annotation: VideoAnnotation) -> pd.DataFrame:
        """Generate an events DataFrame in the same format produced by capture.py.

        Creates one 'neutral'/'segment_end' pair for each neutral segment and
        one 'measurement'/'segment_end' pair for each task repetition. For
        continuous-mode annotations, a single measurement window spanning the
        post-neutral remainder of the video is emitted. Returns a DataFrame
        sorted by timestamp_abs.
        """
        events: List[Dict] = []

        for neutral_seg in annotation.neutral_segments:
            events.append({
                'timestamp_abs': neutral_seg['start_time'],
                'event_type': 'neutral',
                'task_group': None,
                'task_id': None,
                'task_name': None,
            })
            events.append({
                'timestamp_abs': neutral_seg['end_time'],
                'event_type': 'segment_end',
                'task_group': None,
                'task_id': None,
                'task_name': None,
            })

        if not annotation.is_continuous:
            for task in annotation.tasks:
                for rep in task.repetitions:
                    events.append({
                        'timestamp_abs': rep['start_time'],
                        'event_type': 'measurement',
                        'task_group': task.task_group,
                        'task_id': task.task_id,
                        'task_name': task.task_name,
                    })
                    events.append({
                        'timestamp_abs': rep['end_time'],
                        'event_type': 'segment_end',
                        'task_group': task.task_group,
                        'task_id': task.task_id,
                        'task_name': task.task_name,
                    })
        else:
            neutral_end = (
                annotation.neutral_segments[0]['end_time']
                if annotation.neutral_segments
                else 0.0
            )
            events.append({
                'timestamp_abs': neutral_end,
                'event_type': 'measurement',
                'task_group': '0',
                'task_id': 0,
                'task_name': 'continuous',
            })
            events.append({
                'timestamp_abs': annotation.duration_sec,
                'event_type': 'segment_end',
                'task_group': '0',
                'task_id': 0,
                'task_name': 'continuous',
            })

        return pd.DataFrame(events).sort_values('timestamp_abs').reset_index(drop=True)

    def save_annotation_template(self, output_path: Path) -> None:
        """Save a JSON annotation template file for manual editing.

        The template includes the actual video file metadata (fps, duration,
        total_frames) and placeholder entries for neutral segments and tasks.
        Edit the template to supply the actual task timings, then pass it to
        load_annotations() via the --annotations CLI flag.
        """
        template = {
            "_INSTRUCTIONS": "Edit this file with actual task timings, then use with --annotations flag",
            "video_info": {
                "path": str(self.video_path),
                "fps": self.fps,
                "duration_sec": self.duration_sec,
                "total_frames": self.total_frames,
            },
            "is_continuous": False,
            "neutral_segments": [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "_note": "Adjust to actual neutral/baseline timing",
                }
            ],
            "tasks": [
                {
                    "task_group": "A",
                    "task_id": 1,
                    "task_name": "Smiling Broadly",
                    "repetitions": [
                        {"start_time": 6.0, "end_time": 9.0},
                        {"start_time": 10.0, "end_time": 13.0},
                        {"start_time": 14.0, "end_time": 17.0},
                    ],
                    "_note": "Add/edit tasks and repetition timings as needed",
                }
            ],
        }
        save_json(template, output_path)
        print(f"\nAnnotation template saved: {output_path}")
        print("Next steps:")
        print("1. Edit the template with actual task timings")
        print(f"2. Re-run with: --video {self.video_path} --annotations {output_path}")


def create_video_processor(video_path: Path) -> VideoFileProcessor:
    """Create and return a VideoFileProcessor for the given video file path."""
    return VideoFileProcessor(video_path)


def create_multi_camera_processor_from_paths(
    video_paths: List[Path],
    features_config: Dict[str, Any],
    model_path: Path,
):
    """Create a MultiCameraProcessor via a deferred import to avoid circular imports.

    This shim is the preferred entry point when the caller already holds a
    reference to video_processor and needs multi-camera processing. It defers
    the import of multi_camera_processor until call time so that the two
    modules do not form an import cycle.
    """
    from .multi_camera_processor import create_multi_camera_processor
    return create_multi_camera_processor(video_paths, features_config, model_path)
