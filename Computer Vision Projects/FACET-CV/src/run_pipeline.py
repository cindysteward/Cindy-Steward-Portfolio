"""
Main pipeline orchestrator for facial motor and speech behavior analysis.

Coordinates video capture, preprocessing, feature extraction, baseline correction,
metrics computation, anomaly detection, clinical decision support, and visualization
in a single end-to-end pass. Also supports offline processing of pre-recorded video
files with manual annotations, automatic segment detection, or continuous recording mode.

Usage::

    python3 src/run_pipeline.py --mode pilot --subject P001 --session baseline --input live
    python3 src/run_pipeline.py --mode pilot --subject P001 --session test1 --input /path/to/video.mp4
    python3 src/run_pipeline.py --mode patient --subject PAT001 --session intra_op --input live \\
        --reference P001_baseline_20260101_120000
    python3 src/run_pipeline.py --mode patient --subject PAT001 --session pre_op \\
        --video recording.mp4 --annotations recording.json
    python3 src/run_pipeline.py --mode pilot --subject P001 --session test \\
        --video recording.mp4 --auto-detect

Study Prompter recordings (single profile):
  python run_pipeline.py --mode pilot --subject P001 --session test1 \\
    --prompter-videos P001_cam1_*.mp4 \\
    --prompter-timestamps P001_timestamps_*.csv \\
    --prompter-meta P001_recording_meta_*.json

Study Prompter recordings (COMBINED profile, two cameras):
  python run_pipeline.py --mode pilot --subject P001 --session combined1 \\
    --prompter-videos P001_cam1_*.mp4 P001_cam2_*.mp4 \\
    --prompter-timestamps P001_timestamps_*.csv \\
    --prompter-assembly P001_assembly_*.csv \\
    --prompter-meta P001_recording_meta_*.json
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_manager import IOManager
from src.capture import create_capture, CaptureConfig
from src.preprocessing import create_preprocessor
from src.baseline import create_baseline_constructor, create_baseline_corrector
from src.feature_extraction import create_feature_extractor
from src.metrics import create_metrics_computer
from src.anomaly import create_anomaly_detector, create_cusum_monitor, CUSUMMonitor
from src.articulation import create_articulation_scorer
from src.decision_support import create_decision_support
from src.visualization import create_visualizer
from src.validation import create_pilot_validator
from src.consolidate import consolidate_subject
from src.task_profile import TaskProfile, load_task_profile
from src.trends import create_trend_analyzer
from src.anatomy import generate_anatomical_report
from src.utils import (
    setup_logging,
    save_json,
    load_json,
    get_pipeline_version,
    resolve_dominant_task,
)

_NOISE_PSNR_THRESHOLD = 28.0
_MIN_BASELINE_FRAMES = 30
_MIN_CONTINUOUS_WINDOW_FRAMES = 10
_CONTINUOUS_WINDOW_SEC = 60


def _parse_timestamp(timestamp_str: str) -> Optional[float]:
    """Convert a *MM:SS* or *HH:MM:SS* string to total seconds."""
    if not timestamp_str:
        return None
    parts = timestamp_str.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        print(f"Warning: Invalid timestamp format '{timestamp_str}', expected MM:SS or HH:MM:SS")
        return None
    except ValueError:
        print(f"Warning: Could not parse timestamp '{timestamp_str}'")
        return None


def _load_clinical_notes(filepath: str) -> Optional[Dict[str, Any]]:
    """Return parsed JSON from *filepath*, or ``None`` on any failure."""
    try:
        with open(filepath) as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"Warning: Could not load clinical notes from '{filepath}': {exc}")
        return None


def _prompt_for_task() -> Optional[str]:
    """Interactively ask the operator which task was performed in the video."""
    print("\n" + "=" * 50)
    print("TASK SELECTION FOR VIDEO PROCESSING")
    print("=" * 50)
    print("\nTask Groups:")
    print("  A: Non-Speech Facial Expression Tasks")
    print("     1-Pursing lips, 2-Smiling, 3-Showing teeth,")
    print("     4-Tongue out, 5-Tongue to corner, 6-Tongue to lip,")
    print("     7-Frowning, 8-Puffing cheeks, 9-Raising eyebrows")
    print("  B: Speech Articulation Tasks")
    print("     1-Pa-Pa-Pa, 2-Ta-Ta-Ta, 3-Ka-Ka-Ka, 4-Pa-Ta-Ka")
    print("  C: Word Production Tasks")
    print("\nEnter task (e.g., 'A2' for Smiling, 'B1' for Pa-Pa-Pa)")
    print("Or press Enter to skip (no task annotation):")
    print("=" * 50)
    try:
        user_input = input("Task: ").strip().upper()
        if user_input:
            return user_input
    except (EOFError, KeyboardInterrupt):
        pass
    return None


def _merge_articulation_scores(score_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple articulation score dicts by averaging numeric fields.

    Preserves nested dict structure (e.g., per_task_scores, per_word_scores).
    For each numeric field, computes the mean across available dictionaries.
    """
    if not score_dicts:
        return {}
    if len(score_dicts) == 1:
        return score_dicts[0]

    merged: Dict[str, Any] = {}
    all_keys = set()
    for d in score_dicts:
        all_keys.update(d.keys())

    for key in all_keys:
        values = []
        nested_dicts = []
        for d in score_dicts:
            if key in d:
                v = d[key]
                if isinstance(v, (int, float)):
                    values.append(v)
                elif isinstance(v, dict):
                    nested_dicts.append(v)

        if values:
            merged[key] = float(np.mean(values))
        elif nested_dicts:
            merged[key] = _merge_articulation_scores(nested_dicts)
        elif score_dicts[0].get(key) is not None:
            merged[key] = score_dicts[0][key]

    return merged


def process_video_file(
    video_path: Path,
    annotation_path: Optional[Path],
    subject_id: str,
    session_label: str,
    study_mode: str,
    auto_detect: bool = False,
    continuous: bool = False,
) -> None:
    """Process a pre-recorded video file through the full analysis pipeline.

    Modes:
    1. Manual annotations — provide *annotation_path* JSON.
    2. Auto-detect — use motion detection to find segments.
    3. Continuous — treat as non-task-based continuous recording.
    4. Template — no flags generates an annotation template for editing.

    After frame/event generation the standard Pipeline is invoked so every
    study mode (pilot, patient) and session label (baseline, test, pre_op,
    intra_op, post_op, etc.) works identically to live capture.
    """
    from src.video_processor import create_video_processor

    print(f"\n{'=' * 70}")
    print(f"PROCESSING VIDEO FILE: {video_path.name}")
    print(f"{'=' * 70}\n")

    video_proc = create_video_processor(video_path)

    if continuous:
        print("MODE: Continuous (non-task-based) processing")
        annotation = video_proc.create_continuous_annotation()
    elif annotation_path and annotation_path.exists():
        print(f"MODE: Manual annotations from {annotation_path}")
        annotation = video_proc.load_annotations(annotation_path)
    elif auto_detect:
        print("MODE: Auto-detect segments from motion")
        annotation = video_proc.auto_detect_segments()

        auto_path = video_path.with_suffix('.auto_annotations.json')
        save_json({
            'neutral_segments': annotation.neutral_segments,
            'tasks': [
                {
                    'task_group': t.task_group,
                    'task_id': t.task_id,
                    'task_name': t.task_name,
                    'repetitions': t.repetitions,
                }
                for t in annotation.tasks
            ],
        }, auto_path)
        print(f"\nAUTO-DETECTED annotations saved: {auto_path}")
        print(f"REVIEW AND EDIT, then re-run with: --annotations {auto_path}")
        return
    else:
        print("MODE: Creating annotation template")
        template_path = video_path.with_suffix('.annotation_template.json')
        video_proc.save_annotation_template(template_path)
        return

    print("\nGenerating frame data and events...")
    frame_data = video_proc.generate_frame_data(annotation)
    events_df = video_proc.generate_events_df(annotation)

    neutral_frames = sum(1 for f in frame_data if f['segment'] == 'neutral')
    measurement_frames = sum(1 for f in frame_data if f['segment'] == 'measurement')
    print(f"  Total frames: {len(frame_data)}")
    print(f"  Neutral: {neutral_frames}, Measurement: {measurement_frames}")
    print(f"  Events: {len(events_df)}")

    pipeline = Pipeline(study_mode=study_mode, subject_id=subject_id, session_label=session_label)
    logger = pipeline.logger

    logger.info("Processing video file: %s", video_path)

    pd.DataFrame(frame_data).to_csv(pipeline.io.get_frame_data_path(), index=False)
    events_df.to_csv(pipeline.io.get_events_path(), index=False)

    copied_path = pipeline.io.copy_input_video(video_path)

    logger.info("Step 1: Running MediaPipe face detection on video...")
    frame_df, _landmarks_df, blendshapes_df = pipeline.preprocessor.process_video(
        copied_path, frame_data, events_df,
    )

    pipeline.io.save_dataframe(frame_df, pipeline.io.get_frame_data_path())
    pipeline.io.save_dataframe(blendshapes_df, pipeline.io.get_blendshapes_path())
    logger.info(
        "Detected faces in %d/%d frames",
        frame_df['detection_success'].sum(), len(frame_df),
    )

    pipeline.frame_data = frame_data
    pipeline.events_df = events_df
    pipeline.blendshapes_df = blendshapes_df

    logger.info("Step 3: Baseline construction")
    pipeline._construct_baseline()
    pipeline._validate_baseline_quality()

    logger.info("Step 4: Feature extraction with baseline correction")
    pipeline._extract_features()

    logger.info("Step 5: Metrics computation")
    pipeline._compute_metrics()

    if study_mode == "patient":
        logger.info("Step 5b: Computing continuous session metrics")
        pipeline._compute_continuous_metrics()

    logger.info("Step 5d: Articulation scoring")
    pipeline._compute_articulation_scores()
    pipeline._inject_per_rep_scores()

    logger.info("Step 5c: Task profile management")
    pipeline._manage_task_profile()

    logger.info("Step 5e: Enhanced assessment features")
    pipeline._compute_enhanced_assessment_features()

    logger.info("Step 6: Anomaly detection")
    pipeline._detect_anomalies(None)

    logger.info("Step 6b: Anatomical analysis")
    pipeline._run_anatomical_analysis()

    logger.info("Step 6c: Longitudinal trend analysis")
    pipeline._run_trend_analysis()

    logger.info("Step 7: Decision support / screening")
    pipeline._apply_decision_support()

    logger.info("Step 8: Visualization generation")
    pipeline._generate_visualizations()

    logger.info("Step 10: Saving results")
    pipeline._save_results()

    print(f"\n{'=' * 70}")
    print("VIDEO PROCESSING COMPLETE")
    print(f"Results saved to: {pipeline.io.results_dir}")
    print(f"{'=' * 70}\n")


class Pipeline:
    """End-to-end analysis pipeline for a single recording session."""

    def __init__(self, study_mode: str, subject_id: str, session_label: str):
        """Initialise IO paths, logging, config files, and all processing modules."""
        self.study_mode = study_mode
        self.subject_id = subject_id
        self.session_label = session_label

        self.io = IOManager(PROJECT_ROOT, subject_id, session_label, study_mode)
        self.logger = setup_logging(self.io.logs_dir, self.io.session_id)
        self.logger.info("Initializing pipeline v%s", get_pipeline_version())
        self.logger.info("Study mode: %s, Subject: %s, Session: %s", study_mode, subject_id, session_label)

        self.tasks_config = self.io.load_config("tasks")
        self.features_config = self.io.load_config("features")
        self.decision_rules_config = self.io.load_config("decision_rules")
        self.plotting_config = self.io.load_config("plotting")

        self.capture = create_capture(self.plotting_config)
        self.preprocessor = create_preprocessor(self.features_config)
        self.baseline_constructor = create_baseline_constructor(self.features_config)
        self.feature_extractor = create_feature_extractor(self.features_config, self.tasks_config)
        self.metrics_computer = create_metrics_computer(self.features_config, self.tasks_config)
        self.anomaly_detector = create_anomaly_detector(self.decision_rules_config, self.tasks_config)
        self.articulation_scorer = create_articulation_scorer(self.tasks_config)
        self.decision_support = create_decision_support(self.decision_rules_config)
        self.visualizer = create_visualizer(self.plotting_config)
        self.validator = create_pilot_validator(self.tasks_config) if study_mode == "pilot" else None

        self.frame_data: List[Dict] = []
        self.events_df: Optional[pd.DataFrame] = None
        self.blendshapes_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.repetition_metrics_df: Optional[pd.DataFrame] = None
        self.task_metrics_df: Optional[pd.DataFrame] = None
        self.session_metrics: Dict[str, Any] = {}
        self.anomaly_results: Dict[str, Any] = {}
        self.cusum_monitor: Optional[CUSUMMonitor] = None
        self.cusum_results: Optional[Dict[str, Any]] = None
        self.screening_results: Dict[str, Any] = {}
        self.articulation_scores: Dict[str, Any] = {}
        self.reference_articulation_scores: Optional[Dict[str, Any]] = None

        self.reference_session_id: Optional[str] = None
        self.reference_session_ids: Optional[List[str]] = None
        self.reference_baseline_stats: Optional[Dict] = None
        self.is_baseline_session: bool = "baseline" in session_label.lower()
        self.task_profile: Optional[TaskProfile] = None
        self.task_profile_ref: Optional[Dict] = None

    def run(
        self,
        input_source: str,
        reference_session: Optional[str] = None,
        alteration_type: Optional[str] = None,
        task_info: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        clinical_notes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute all pipeline stages and return a summary dict.

        *reference_session* may be a single session ID string, a list of session IDs,
        or None. If multiple references are provided, their articulation scores are
        merged (field-wise averaged) for comparison.
        """
        self.clinical_notes = clinical_notes

        if reference_session is None:
            self.reference_session_ids = None
            self.reference_session_id = None
        elif isinstance(reference_session, str):
            self.reference_session_ids = [reference_session]
            self.reference_session_id = reference_session
        elif isinstance(reference_session, list):
            self.reference_session_ids = reference_session
            self.reference_session_id = reference_session[0] if reference_session else None
        else:
            self.reference_session_ids = None
            self.reference_session_id = None

        try:
            self.logger.info("=" * 60)
            self.logger.info("PIPELINE EXECUTION STARTED")
            self.logger.info("=" * 60)

            self.logger.info("Step 1: Video capture/processing")
            self._capture_video(input_source, task_info, start_time, end_time)

            self.logger.info("Step 2: Preprocessing and feature extraction")
            self._preprocess()

            self.logger.info("Step 3: Baseline construction")
            self._construct_baseline()
            self._validate_baseline_quality()

            self.logger.info("Step 4: Feature extraction with baseline correction")
            self._extract_features()

            self.logger.info("Step 5: Metrics computation")
            self._compute_metrics()

            if self.study_mode == "patient":
                self.logger.info("Step 5b: Computing continuous session metrics (fatigue, trends)")
                self._compute_continuous_metrics()

            self.logger.info("Step 5d: Articulation scoring")
            self._compute_articulation_scores()
            self._inject_per_rep_scores()

            self.logger.info("Step 5c: Task profile management")
            self._manage_task_profile()

            self.logger.info("Step 5e: Enhanced assessment features")
            self._compute_enhanced_assessment_features()

            self.logger.info("Step 6: Anomaly detection")
            self._detect_anomalies(reference_session)

            self.logger.info("Step 6b: Anatomical analysis")
            self._run_anatomical_analysis()

            self.logger.info("Step 6c: Longitudinal trend analysis")
            self._run_trend_analysis()

            self.logger.info("Step 7: Decision support / screening")
            self._apply_decision_support()

            if clinical_notes:
                self.logger.info("Step 7b: Comparing with clinical notes")
                self._compare_clinical_notes(clinical_notes)

            self.logger.info("Step 8: Visualization generation")
            self._generate_visualizations()

            if self.study_mode == "pilot" and alteration_type:
                self.logger.info("Step 9: Pilot study validation")
                self._run_validation(alteration_type)

            self.logger.info("Step 10: Saving results")
            self._save_results()

            self.logger.info("=" * 60)
            self.logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            return self._get_summary()

        except Exception as exc:
            self.logger.error("Pipeline execution failed: %s", exc, exc_info=True)
            raise

    def _capture_video(
        self,
        input_source: str,
        task_info: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """Acquire frames from a live camera or a pre-recorded video file."""
        if input_source.lower() == "live":
            self.logger.info("Starting live video capture...")
            self.frame_data, events = self.capture.capture_live(
                self.io.get_raw_video_path(),
                self.io.get_annotated_video_path(),
                self.io.get_normal_speed_video_path(),
                self.io.get_normal_speed_annotated_video_path(),
            )
            self.events_df = self.capture.get_events_dataframe()
            self.logger.info("Captured %d frames, %d events", len(self.frame_data), len(events))
        else:
            video_path = Path(input_source)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            self.logger.info("Processing video file: %s", video_path)
            if task_info:
                self.logger.info("Task specified: %s", task_info)
            if start_time is not None or end_time is not None:
                self.logger.info("Time range: %ss to %ss", start_time, end_time)
            copied_path = self.io.copy_input_video(video_path)
            self.frame_data, events = self.capture.process_video_file(
                copied_path,
                self.io.get_annotated_video_path(),
                output_normal_video_path=self.io.get_normal_speed_video_path(),
                task_info=task_info,
                start_time=start_time,
                end_time=end_time,
            )
            self.events_df = self.capture.get_events_dataframe()
            self.logger.info("Processed %d frames", len(self.frame_data))

        if self.events_df is not None and len(self.events_df) > 0:
            self.io.save_dataframe(self.events_df, self.io.get_events_path())

    def _preprocess(self) -> None:
        """Run face detection, landmark extraction, and blendshape computation."""
        video_path = self.io.get_annotated_video_path()
        if not video_path.exists():
            video_path = self.io.get_raw_video_path()
        if not video_path.exists():
            self.logger.warning("No video file found for preprocessing, using frame data")
            self.blendshapes_df = pd.DataFrame(self.frame_data)
            return

        frame_df, _landmarks_df, blendshapes_df = self.preprocessor.process_video(
            video_path, self.frame_data, self.events_df
        )
        self.io.save_dataframe(frame_df, self.io.get_frame_data_path())
        self.io.save_dataframe(blendshapes_df, self.io.get_blendshapes_path())
        self.blendshapes_df = blendshapes_df

        quality_stats = self.preprocessor.get_detection_quality_stats(frame_df)
        self.logger.info(
            "Detection quality: %.2f%%, Mean PSNR: %.2f dB",
            quality_stats["detection_rate"] * 100,
            quality_stats["mean_psnr"],
        )
        if quality_stats["mean_psnr"] < _NOISE_PSNR_THRESHOLD:
            self.logger.warning(
                "Mean PSNR is low (%.2f dB) — results may be low-confidence due to noise.",
                quality_stats["mean_psnr"],
            )
            self.session_metrics["low_confidence_due_to_noise"] = True

    def _construct_baseline(self) -> None:
        """Load a reference baseline or compute one from this session's neutral frames.

        For test sessions a reference baseline is loaded and used for z-scoring so
        that features are expressed relative to the participant's own neutral.

        For reference/baseline sessions the baseline is always computed from this
        session's own neutral segment (so the task profile captures metrics
        expressed in this session's coordinate frame).  If a prior reference session
        is provided it is loaded *separately* into ``reference_baseline_stats`` for
        visualisation comparison only — it does not override the z-scoring baseline.
        """
        if self.blendshapes_df is None or len(self.blendshapes_df) == 0:
            self.logger.warning("No blendshape data available for baseline construction")
            return

        if self.reference_session_id and not self.is_baseline_session:
            ref_subject = self.reference_session_id.split("_")[0] or self.subject_id
            ref_baseline_path = (
                self.io.data_dir / "raw" / self.study_mode / ref_subject
                / self.reference_session_id / "baseline.json"
            )
            if ref_baseline_path.exists():
                try:
                    self.baseline_constructor.load_baseline(ref_baseline_path)
                    self.reference_baseline_stats = self.baseline_constructor.baseline_stats
                    self.logger.info("Loaded baseline statistics from reference session: %s", ref_baseline_path)
                    return
                except Exception as exc:
                    self.logger.warning(
                        "Could not load baseline from reference session %s: %s",
                        self.reference_session_id, exc,
                    )
        elif self.reference_session_id and self.is_baseline_session:
            ref_subject = self.reference_session_id.split("_")[0] or self.subject_id
            ref_baseline_path = (
                self.io.data_dir / "raw" / self.study_mode / ref_subject
                / self.reference_session_id / "baseline.json"
            )
            if ref_baseline_path.exists():
                try:
                    prior_stats = load_json(ref_baseline_path)
                    self.reference_baseline_stats = prior_stats.get("stats", prior_stats)
                    self.logger.info(
                        "Loaded prior reference baseline for comparison: %s", ref_baseline_path
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Could not load prior baseline from reference session %s: %s",
                        self.reference_session_id, exc,
                    )

        neutral_df = self.blendshapes_df[self.blendshapes_df["segment"] == "neutral"]
        if len(neutral_df) == 0:
            self.logger.warning("No explicit neutral segment found, using first 10%% of data as baseline")
            n_baseline = max(_MIN_BASELINE_FRAMES, int(len(self.blendshapes_df) * 0.1))
            neutral_df = self.blendshapes_df.head(n_baseline)

        baseline_data = self.baseline_constructor.compute_baseline(self.blendshapes_df, neutral_df)
        self.baseline_constructor.compute_observed_ranges(self.blendshapes_df)
        try:
            self.baseline_constructor.save_baseline(self.io.get_baseline_path())
        except Exception:
            pass
        self.logger.info("Baseline computed from %d frames", baseline_data["metadata"]["n_frames"])

    def _extract_features(self) -> None:
        """Standardise blendshapes against baseline and extract per-frame features."""
        if self.blendshapes_df is None:
            return
        corrector = create_baseline_corrector(self.baseline_constructor)
        standardized_df = corrector.standardize_features(self.blendshapes_df)
        self.features_df = self.feature_extractor.extract_features(
            standardized_df, self.events_df,
            baseline_stats=self.baseline_constructor.baseline_stats,
            observed_ranges=self.baseline_constructor.observed_ranges
        )
        self.io.save_dataframe(self.features_df, self.io.get_corrected_features_path())
        self.logger.info("Extracted %d features (standardized)", len(self.features_df.columns))

    def _compute_metrics(self) -> None:
        """Aggregate features into repetition-level, task-level, and session-level metrics."""
        if self.features_df is None:
            return

        self.repetition_metrics_df = self.metrics_computer.compute_repetition_metrics(self.features_df)

        if "repetition" in self.repetition_metrics_df.columns and len(self.repetition_metrics_df) > 0:
            rep_ids = sorted(int(r) for r in self.repetition_metrics_df["repetition"].unique() if r != 0)
            if rep_ids and rep_ids != list(range(rep_ids[0], rep_ids[0] + len(rep_ids))):
                self.logger.warning(
                    "Non-contiguous repetition IDs detected: %s. Check for missed 'm'/'r' presses.", rep_ids
                )

        if len(self.repetition_metrics_df) > 0:
            reps_to_save = self.repetition_metrics_df[self.repetition_metrics_df["repetition"] != 0].copy()
            self.io.save_dataframe(reps_to_save, self.io.get_repetition_metrics_path())

        self.task_metrics_df = self.metrics_computer.compute_task_metrics(self.repetition_metrics_df)
        if len(self.task_metrics_df) > 0:
            self.io.save_dataframe(self.task_metrics_df, self.io.get_task_metrics_path())

        self.session_metrics = self.metrics_computer.compute_session_metrics(
            self.task_metrics_df, self.repetition_metrics_df
        )
        save_json(self.session_metrics, self.io.get_session_metrics_path())
        self.logger.info("Computed metrics for %d repetitions", len(self.repetition_metrics_df))

    def _manage_task_profile(self) -> None:
        """Load or create a task profile, update it if this is a baseline session."""
        profile_path = self.io.get_task_profile_path()
        profile_path.parent.mkdir(parents=True, exist_ok=True)

        self.task_profile = load_task_profile(profile_path, self.subject_id)
        if self.task_profile is None:
            self.task_profile = TaskProfile(self.subject_id)

        if self.is_baseline_session and self.repetition_metrics_df is not None and len(self.repetition_metrics_df) > 0:
            self.task_profile.update_from_session(
                self.io.session_id,
                self.repetition_metrics_df,
                features_df=self.features_df,
                task_metrics_df=self.task_metrics_df,
            )
            self.task_profile.save(profile_path)
            self.logger.info(
                "Task profile updated: %d sessions, %d task(s)",
                len(self.task_profile.sessions_included),
                len(self.task_profile.tasks),
            )

        if self.task_profile.is_loaded():
            task_group, task_id = resolve_dominant_task(self.frame_data)
            self.task_profile_ref = self.task_profile.get_task_reference(task_group, task_id)
            if self.task_profile_ref:
                self.logger.info(
                    "Task profile loaded for %s_%s: %d session(s), %d repetition(s)",
                    task_group, task_id,
                    self.task_profile_ref.get("n_sessions", 0),
                    self.task_profile_ref.get("n_repetitions_total", 0),
                )

    def _compute_articulation_scores(self) -> None:
        """Compute articulation quality scores for speech tasks (Groups B and C).

        When a reference session is available its articulation scores are loaded
        so that downstream decision-support can evaluate deviation from the
        subject's own baseline rather than relying on absolute thresholds.
        """
        if self.repetition_metrics_df is None or len(self.repetition_metrics_df) == 0:
            return

        has_speech_tasks = False
        if "task_group" in self.repetition_metrics_df.columns:
            has_speech_tasks = self.repetition_metrics_df["task_group"].isin(
                ["B", "C"]
            ).any()

        if not has_speech_tasks:
            return

        self._load_reference_articulation()

        self.articulation_scores = self.articulation_scorer.compute_scores(
            self.repetition_metrics_df, self.features_df,
            reference_articulation=self.reference_articulation_scores,
            session_id=self.io.session_id,
        )

        word_prod = self.articulation_scorer.compute_word_production_features(
            self.articulation_scores.get("per_task_scores", {})
        )
        if word_prod:
            self.articulation_scores.update(word_prod)

        for key in (
            "articulation_score_pataka",
            "simple_syllable_mean",
            "mean_articulation_score",
            "group_b_articulation_score",
            "group_c_articulation_score",
            "articulation_impairment_consistency",
            "articulation_score_pa",
            "articulation_score_ta",
            "articulation_score_ka",
            "word_production_quality",
            "complexity_gradient",
            "cross_word_consistency",
            "word_production_impairment_rate",
            "ors_gravity_flag",
            "group_b_timing_mean", "group_b_smoothness_mean", "group_b_amplitude_mean",
            "group_c_timing_mean", "group_c_smoothness_mean", "group_c_amplitude_mean",
            "group_b_timing_deviation", "group_b_smoothness_deviation", "group_b_amplitude_deviation",
            "group_c_timing_deviation", "group_c_smoothness_deviation", "group_c_amplitude_deviation",
            "group_b_n_timing_drop", "group_b_n_smoothness_drop", "group_b_n_amplitude_drop",
            "group_c_n_timing_drop", "group_c_n_smoothness_drop", "group_c_n_amplitude_drop",
            "b4_simple_act_ratio", "b4_simple_act_ratio_vs_ref",
            "n_c_complex_extreme_amp_drop",
            "group_b_mean_duration_ratio",
            "group_b_n_slow_tasks",
            "group_b_n_fast_tasks",
        ):
            if key in self.articulation_scores:
                self.session_metrics[key] = self.articulation_scores[key]

        save_json(self.articulation_scores, self.io.results_dir / "articulation_scores.json")
        self.logger.info(
            "Articulation scoring: %d tasks, mean=%.2f, consistency=%.2f",
            self.articulation_scores.get("n_tasks_scored", 0),
            self.articulation_scores.get("mean_articulation_score", 0),
            self.articulation_scores.get("articulation_impairment_consistency", 0),
        )
        if word_prod:
            self.logger.info(
                "Word production: %d words, quality=%.2f, gradient=%+.2f, consistency=%.2f",
                word_prod.get("n_words_scored", 0),
                word_prod.get("word_production_quality", 0),
                word_prod.get("complexity_gradient", 0),
                word_prod.get("cross_word_consistency", 0),
            )
        if self.reference_articulation_scores:
            ref_mean = self.reference_articulation_scores.get("mean_articulation_score", 0)
            cur_mean = self.articulation_scores.get("mean_articulation_score", 0)
            self.logger.info(
                "Articulation baseline comparison: ref=%.2f, test=%.2f, delta=%+.3f",
                ref_mean, cur_mean, cur_mean - ref_mean,
            )

    def _inject_per_rep_scores(self) -> None:
        """Annotate repetition_metrics_df with per-rep articulation component scores.

        Must be called after _compute_articulation_scores() and before
        _detect_anomalies().  Injects four columns derived from the same
        scoring logic used by ArticulationScorer for task-level scores:

        ==========================  =====================  ========================
        Column                      Anomaly keyword group  Disorder distinction
        ==========================  =====================  ========================
        kinematic_smoothness        kinematic_profile      tremor, motor noise
        rep_temporal_consistency    articulation           variability → apraxia
        rep_spatial_consistency     articulation           groping → apraxia
        rep_articulation_score      articulation           overall rep quality
        ==========================  =====================  ========================

        Scores are normalised within-task (intra-session) so they capture
        rep-to-rep VARIABILITY — the apraxia/groping signal — while the
        existing raw feature z-scores (duration_sec, jawOpen_range) continue
        to capture ABSOLUTE deviations from the reference session baseline.
        Only B and C task rows receive values; other rows remain NaN.

        After injection the updated DataFrame is saved back to
        ``repetition_metrics.csv`` so that when this session is later used
        as a reference the anomaly detector's ``fit()`` sees the new columns.
        This method must run BEFORE ``_manage_task_profile()`` so that the
        task profile's ``_raw_values`` also include the new columns.
        """
        if (
            self.features_df is None
            or self.repetition_metrics_df is None
            or len(self.repetition_metrics_df) == 0
        ):
            return

        needed_cols = {"task_group", "task_id", "repetition"}
        if not needed_cols.issubset(self.repetition_metrics_df.columns):
            return

        per_rep = self.articulation_scorer.compute_per_rep_scores(self.features_df)
        if not per_rep:
            return

        score_cols = [
            "kinematic_smoothness",
            "rep_temporal_consistency",
            "rep_spatial_consistency",
            "rep_articulation_score",
        ]
        arrays = {
            col: np.full(len(self.repetition_metrics_df), np.nan)
            for col in score_cols
        }

        for i, row in enumerate(self.repetition_metrics_df.itertuples(index=False)):
            key = (str(row.task_group), int(row.task_id), int(row.repetition))
            entry = per_rep.get(key)
            if entry is None:
                continue
            for col in score_cols:
                val = entry.get(col)
                if val is not None:
                    arrays[col][i] = val

        self.repetition_metrics_df = self.repetition_metrics_df.copy()
        for col in score_cols:
            self.repetition_metrics_df[col] = arrays[col]

        n_filled = int(np.isfinite(arrays["rep_articulation_score"]).sum())
        self.logger.debug(
            "Injected per-rep articulation scores (timing/smoothness/amplitude/composite) "
            "for %d/%d repetitions",
            n_filled, len(self.repetition_metrics_df),
        )

        try:
            reps_to_save = self.repetition_metrics_df[
                self.repetition_metrics_df["repetition"] != 0
            ].copy() if "repetition" in self.repetition_metrics_df.columns else self.repetition_metrics_df.copy()
            self.io.save_dataframe(reps_to_save, self.io.get_repetition_metrics_path())
        except Exception as _exc:
            self.logger.warning("Could not persist per-rep scores to CSV: %s", _exc)

    def _load_reference_articulation(self) -> None:
        """Load and merge baseline articulation scores from reference session(s).

        If multiple reference sessions are provided, their articulation_scores.json
        files are loaded and merged (field-wise averaged) to produce a single
        reference profile for comparison.

        For a new reference session that already has prior reference data (i.e. this
        is a second baseline), the prior reference articulation is still loaded so
        the visualizations can compare the two reference sessions.
        """
        if not self.reference_session_ids:
            return

        score_dicts = []
        for ref_id in self.reference_session_ids:
            parts = ref_id.split("_")
            ref_subject = parts[0] if len(parts) >= 2 else self.subject_id
            ref_artic_path = (
                self.io.data_dir / "results" / self.study_mode
                / ref_subject / ref_id / "articulation_scores.json"
            )

            if ref_artic_path.exists():
                try:
                    score_dict = load_json(ref_artic_path)
                    score_dicts.append(score_dict)
                    self.logger.info("Loaded reference articulation scores from %s", ref_id)
                except Exception as exc:
                    self.logger.warning(
                        "Could not load reference articulation scores from %s: %s",
                        ref_id, exc
                    )
            else:
                self.logger.info(
                    "No reference articulation scores found at %s", ref_artic_path
                )

        if score_dicts:
            self.reference_articulation_scores = _merge_articulation_scores(score_dicts)
            if len(self.reference_session_ids) > 1:
                self.logger.info(
                    "Merged articulation scores from %d reference sessions",
                    len(score_dicts)
                )

    def _compute_enhanced_assessment_features(self) -> None:
        """Compute cross-task matching, enhanced speech features, and pattern analysis.

        Adds features to session_metrics for improved disorder differentiation
        in the decision support module.  Cross-task matching compares each test
        repetition against all reference task profiles in the same group to
        detect task substitution (buccofacial apraxia).  Enhanced speech
        features include duration ratios, repetition variability, and
        cross-group comparisons for dysarthria, speech apraxia, and
        phonological disorder differentiation.
        """
        if self.task_profile is None or not self.task_profile.is_loaded():
            return
        if self.repetition_metrics_df is None or len(self.repetition_metrics_df) == 0:
            return

        task_group, task_id = resolve_dominant_task(self.frame_data)
        has_task_cols = "task_group" in self.repetition_metrics_df.columns

        if has_task_cols:
            a_task_ids = self.repetition_metrics_df.loc[
                self.repetition_metrics_df["task_group"] == "A", "task_id"
            ].unique()
        else:
            a_task_ids = [task_id] if task_group == "A" else []

        all_cross_results: Dict[str, Any] = {}
        total_subs = 0
        total_reps = 0
        all_sims: List[float] = []

        for tid in a_task_ids:
            tid = int(tid)
            if has_task_cols:
                task_reps = self.repetition_metrics_df[
                    (self.repetition_metrics_df["task_group"] == "A")
                    & (self.repetition_metrics_df["task_id"] == tid)
                ]
            else:
                task_reps = self.repetition_metrics_df

            _a_task_name: Optional[str] = None
            if "task_name" in task_reps.columns:
                _names = task_reps["task_name"].dropna().unique()
                _names = [n for n in _names if n and str(n) != "(no task selected)"]
                if _names:
                    _a_task_name = str(_names[0])

            matching = self.task_profile.compute_cross_task_matching(
                task_reps, "A", tid, task_name=_a_task_name
            )
            if matching:
                all_cross_results[f"A_{tid}"] = matching
                if 1 <= tid <= 9:
                    total_subs += matching.get("n_substitutions", 0)
                    total_reps += matching.get("n_repetitions_evaluated", 0)
                    all_sims.append(matching.get("task_profile_similarity", 1.0))

        if all_cross_results:
            self.session_metrics["substitution_rate"] = (
                total_subs / total_reps if total_reps > 0 else 0.0
            )
            self.session_metrics["task_profile_similarity"] = (
                float(np.mean(all_sims)) if all_sims else 1.0
            )
            self.session_metrics["mean_substitution_score"] = float(np.mean([
                r.get("mean_substitution_score", 0.0)
                for r in all_cross_results.values()
            ]))
            save_json(all_cross_results, self.io.results_dir / "cross_task_matching.json")
            self.logger.info(
                "Cross-task matching: %d/%d substitutions, mean similarity=%.2f",
                total_subs, total_reps,
                self.session_metrics["task_profile_similarity"],
            )

        if has_task_cols and self.features_df is not None:
            b_task_ids = self.repetition_metrics_df.loc[
                self.repetition_metrics_df["task_group"] == "B", "task_id"
            ].unique()
            pattern_corrs: List[float] = []
            for b_tid in b_task_ids:
                pattern_sim = self.task_profile.compute_activation_pattern_similarity(
                    self.features_df, "B", int(b_tid)
                )
                if pattern_sim and "mean_pattern_correlation" in pattern_sim:
                    pattern_corrs.append(pattern_sim["mean_pattern_correlation"])
            if pattern_corrs:
                self.session_metrics["mean_pattern_correlation"] = float(
                    np.mean(pattern_corrs)
                )

        enhanced = self.articulation_scorer.compute_enhanced_speech_features(
            self.repetition_metrics_df, self.task_profile
        )
        for key, value in enhanced.items():
            if isinstance(value, (int, float)):
                self.session_metrics[key] = value

        exec_score = self.metrics_computer.compute_execution_correctness_score(
            self.repetition_metrics_df
        )
        self.session_metrics["execution_correctness_score"] = exec_score

        if enhanced:
            self.logger.info(
                "Enhanced features: duration_ratio=%.2f, B-C dissociation=%.2f, "
                "word_consistency=%.2f",
                enhanced.get("speech_duration_ratio_mean", 1.0),
                enhanced.get("group_bc_dissociation", 0.0),
                enhanced.get("word_cross_rep_consistency_mean", 0.8),
            )

    def _detect_anomalies(self, reference_session: Optional[str | list] = None) -> None:
        """Fit and run anomaly detection independently for each task.

        When a session contains multiple tasks, each (task_group, task_id) is
        analysed against its own reference distribution from the task profile.
        Per-task results are merged into a single anomaly_results dict whose
        parallel lists preserve the original repetition order.
        """
        if self.repetition_metrics_df is None or len(self.repetition_metrics_df) == 0:
            self.anomaly_results = {"summary": {"n_samples": 0, "n_anomalies": 0}}
            return

        if isinstance(reference_session, list):
            reference_session = reference_session[0] if reference_session else None
        if reference_session is not None:
            self.reference_session_id = reference_session

        has_multi_task = (
            "task_group" in self.repetition_metrics_df.columns
            and "task_id" in self.repetition_metrics_df.columns
        )

        if self.is_baseline_session:
            task_group, task_id = resolve_dominant_task(self.frame_data)
            self.anomaly_detector.fit(
                self.repetition_metrics_df, task_group=task_group, task_id=task_id
            )
            self.anomaly_results = self.anomaly_detector.detect_anomalies(
                self.repetition_metrics_df
            )
            self.anomaly_results["baseline_self_check"] = True
            self._add_task_names_to_results()
            save_json(self.anomaly_results, self.io.get_anomaly_results_path())
            self.logger.info(
                "Baseline self-consistency check: %d anomalies",
                self.anomaly_results.get("summary", {}).get("n_anomalies", 0),
            )
            return

        if has_multi_task:
            task_keys = (
                self.repetition_metrics_df
                .groupby(["task_group", "task_id"])
                .size()
                .index.tolist()
            )
        else:
            tg, tid = resolve_dominant_task(self.frame_data)
            task_keys = [(tg, tid)]

        per_task_results: List[Dict[str, Any]] = []
        original_indices: List[int] = []

        for tg, tid in task_keys:
            tid = int(tid)
            if has_multi_task:
                mask = (
                    (self.repetition_metrics_df["task_group"] == tg)
                    & (self.repetition_metrics_df["task_id"] == tid)
                )
                task_df = self.repetition_metrics_df[mask].copy()
                task_idx = list(self.repetition_metrics_df.index[mask])
            else:
                task_df = self.repetition_metrics_df.copy()
                task_idx = list(self.repetition_metrics_df.index)

            if len(task_df) == 0:
                continue

            detector = create_anomaly_detector(
                self.decision_rules_config, self.tasks_config
            )
            detector.task_group = tg
            detector.task_id = int(tid)
            detector._task_config = detector._resolve_task_config(tg, int(tid))

            fitted = self._fit_task_detector(detector, tg, tid, task_df, reference_session)

            if not fitted:
                detector.fit(task_df, task_group=tg, task_id=tid)

            task_results = detector.detect_anomalies(task_df)
            per_task_results.append(task_results)
            original_indices.extend(task_idx)
            self.logger.info(
                "Task %s_%d: %d/%d anomalies (model: %s, n_ref: %d, mean_score: %.3f)",
                tg, tid,
                task_results.get("summary", {}).get("n_anomalies", 0),
                len(task_df),
                task_results.get("model_type", "?"),
                task_results.get("n_reference", 0),
                task_results.get("summary", {}).get("mean_deviation_score", 0),
            )

        self.anomaly_results = self._merge_per_task_results(
            per_task_results, original_indices
        )
        self._add_task_names_to_results()

        lbl = self.session_label.lower()
        is_continuous = lbl.startswith("intra") or lbl == "post_op"
        if is_continuous:
            for tg, tid in task_keys:
                self._run_cusum_monitoring(tg, int(tid))
        for tg, tid in task_keys:
            if self._dtw_available(tg, int(tid)):
                self._run_dtw_analysis(tg, int(tid))

        save_json(self.anomaly_results, self.io.get_anomaly_results_path())
        self.logger.info(
            "Detected %d/%d anomalies across %d tasks (mean score: %.3f)",
            self.anomaly_results.get("summary", {}).get("n_anomalies", 0),
            self.anomaly_results.get("summary", {}).get("n_samples", 0),
            len(task_keys),
            self.anomaly_results.get("summary", {}).get("mean_deviation_score", 0),
        )

    def _fit_task_detector(
        self,
        detector: Any,
        task_group: str,
        task_id: int,
        task_df: pd.DataFrame,
        reference_session: Optional[str],
    ) -> bool:
        """Fit an anomaly detector for a single task using best available reference."""
        _task_name: Optional[str] = None
        if "task_name" in task_df.columns:
            _names = task_df["task_name"].dropna().unique()
            _names = [n for n in _names if n and str(n) != "(no task selected)"]
            if _names:
                _task_name = str(_names[0])

        if self.task_profile is not None and self.task_profile.is_loaded():
            profile_stats = self.task_profile.get_task_feature_stats(
                task_group, task_id, task_name=_task_name
            )
            if profile_stats:
                task_ref = self.task_profile.get_task_reference(
                    task_group, task_id, task_name=_task_name
                )
                if task_ref and task_ref.get("n_sessions", 0) >= 1:
                    detector.set_task_feature_weights(task_ref)
                    if task_ref.get("_is_mapped_reference"):
                        self.logger.info(
                            "Task %s_%s → reference %s_%s via coupling map (%s)",
                            task_group, task_id,
                            task_ref.get("_ref_task_group"), task_ref.get("_ref_task_id"),
                            _task_name or "?",
                        )
                ref_df = self.task_profile.get_reference_metrics_df(
                    task_group, task_id, task_name=_task_name
                )
                detector.fit_from_task_profile(
                    profile_stats, ref_df if len(ref_df) > 0 else None
                )
                return True

        if reference_session:
            ref_path = self.io.get_reference_session_path(reference_session)
            ref_metrics_path = ref_path / "repetition_metrics.csv"
            if ref_metrics_path.exists():
                ref_df = pd.read_csv(ref_metrics_path)
                _DISORDER_CANONICAL = {
                    10: 1, 11: 3, 12: 3, 13: 5, 14: 5, 15: 7, 16: 8, 17: 9,
                }
                ref_tg = task_group
                ref_tid = _DISORDER_CANONICAL.get(task_id, task_id) if task_group == "A" else task_id
                if "task_group" in ref_df.columns and "task_id" in ref_df.columns:
                    ref_df = ref_df[
                        (ref_df["task_group"] == ref_tg)
                        & (ref_df["task_id"].astype(int) == ref_tid)
                    ]
                if len(ref_df) >= 2:
                    detector.fit(ref_df, task_group=ref_tg, task_id=ref_tid)
                    return True

        return False

    @staticmethod
    def _merge_per_task_results(
        per_task_results: List[Dict[str, Any]],
        original_indices: List[int],
    ) -> Dict[str, Any]:
        """Merge per-task anomaly results into one dict preserving original row order."""
        list_keys = [
            "anomaly_scores", "is_anomaly", "deviations", "deviation_score",
            "score_confidence", "anomaly_type", "contributing_features",
            "mahalanobis_score", "centroid_score", "within_session_score",
            "method_votes", "weighted_votes", "method_sigmoid_scores",
            "method_weighted_components", "mahalanobis_ci_lower",
            "mahalanobis_ci_upper", "deviation_ci_lower", "deviation_ci_upper",
            "repetitions", "task_groups", "task_ids",
        ]

        flat: Dict[str, List] = {k: [] for k in list_keys}
        all_feat_devs: Dict[str, Dict] = {}

        for tr in per_task_results:
            for key in list_keys:
                if key in tr:
                    flat[key].extend(tr[key])
            for feat, finfo in tr.get("feature_deviations", {}).items():
                if feat not in all_feat_devs:
                    all_feat_devs[feat] = finfo
                else:
                    existing = all_feat_devs[feat]
                    existing["n_deviant"] = existing.get("n_deviant", 0) + finfo.get("n_deviant", 0)

        sort_order = np.argsort(original_indices).tolist()
        for key in list_keys:
            if key in flat and len(flat[key]) == len(sort_order):
                flat[key] = [flat[key][i] for i in sort_order]

        n_total = len(flat.get("is_anomaly", []))
        n_anom = sum(flat.get("is_anomaly", []))
        dev_scores = flat.get("deviation_score", [])
        conf_scores = flat.get("score_confidence", [])

        merged: Dict[str, Any] = {}
        for key in list_keys:
            merged[key] = flat.get(key, [])
        merged["feature_deviations"] = all_feat_devs
        merged["summary"] = {
            "n_samples": n_total,
            "n_anomalies": n_anom,
            "anomaly_rate": n_anom / n_total if n_total > 0 else 0.0,
            "mean_deviation_score": float(np.mean(dev_scores)) if dev_scores else 0.0,
            "mean_score_confidence": float(np.mean(conf_scores)) if conf_scores else 0.0,
            "n_tasks_analysed": len(per_task_results),
        }
        if per_task_results:
            merged["model_type"] = per_task_results[0].get("model_type", "unknown")
            merged["n_reference"] = per_task_results[0].get("n_reference", 0)
            merged["ml_metadata"] = per_task_results[0].get("ml_metadata", {})
        return merged

    def _add_task_names_to_results(self) -> None:
        """Populate the task_names list in anomaly_results from repetition_metrics_df."""
        if self.repetition_metrics_df is None or "task_name" not in self.repetition_metrics_df.columns:
            return
        names = self.repetition_metrics_df["task_name"].fillna("").tolist()
        self.anomaly_results["task_names"] = names

    def _run_cusum_monitoring(self, task_group: str, task_id: int) -> None:
        """Run CUSUM drift detection for continuous sessions against baseline reference."""
        profile_stats = self.task_profile.get_task_feature_stats(task_group, task_id) if self.task_profile else {}
        if not profile_stats:
            return

        self.cusum_monitor = create_cusum_monitor(k=0.5, h=5.0)
        for feat, stats in profile_stats.items():
            self.cusum_monitor.set_reference(stats.get("mean", 0.0), stats.get("std", 1.0), feat)

        cusum_df = self.cusum_monitor.update_batch(self.repetition_metrics_df)
        if len(cusum_df) > 0:
            alarm_cols = [c for c in cusum_df.columns if c.endswith("_alarm") and c != "any_alarm"]
            n_alarms = int(cusum_df["any_alarm"].sum()) if "any_alarm" in cusum_df.columns else 0
            alarm_features = [c.replace("_alarm", "") for c in alarm_cols
                              if cusum_df[c].any()]
            self.cusum_results = {
                "n_alarms": n_alarms,
                "alarm_features": alarm_features,
                "cusum_state": self.cusum_monitor.get_state(),
            }
            self.anomaly_results["continuous_drift"] = self.cusum_results
            self.logger.info("CUSUM monitoring: %d drift alarms across %d features",
                             n_alarms, len(alarm_features))

    def _dtw_available(self, task_group: str, task_id: int) -> bool:
        """Check whether DTW analysis can be run for the given task."""
        if self.task_profile is None or not self.task_profile.is_loaded():
            return False
        ref = self.task_profile.get_task_reference(task_group, task_id)
        if ref is None:
            return False
        patterns = ref.get("activation_pattern", {})
        return any("curves" in p and len(p.get("curves", [])) >= 2 for p in patterns.values())

    def _run_dtw_analysis(self, task_group: str, task_id: int) -> None:
        """Compute DTW pattern deviation for each test repetition against reference curves."""
        ref = self.task_profile.get_task_reference(task_group, task_id)
        if ref is None or self.features_df is None:
            return

        patterns = ref.get("activation_pattern", {})
        feature = "mean_activation"
        if feature not in patterns or "curves" not in patterns[feature]:
            for f in patterns:
                if "curves" in patterns[f] and len(patterns[f]["curves"]) >= 2:
                    feature = f
                    break
            else:
                return

        ref_curves = [np.array(c) for c in patterns[feature]["curves"]]
        if len(ref_curves) < 2:
            return

        repetitions = sorted(r for r in self.features_df["repetition"].unique() if r != 0)
        dtw_results = []
        for rep in repetitions:
            rep_df = self.features_df[self.features_df["repetition"] == rep]
            if len(rep_df) < 5 or feature not in rep_df.columns:
                dtw_results.append({"repetition": int(rep), "mean_dtw": 0.0, "min_dtw": 0.0, "is_shape_anomaly": False})
                continue
            vals = rep_df[feature].values
            n_bins = len(ref_curves[0])
            t_norm = np.linspace(0, 1, len(vals))
            bins = np.linspace(0, 1, n_bins)
            test_binned = np.interp(bins, t_norm, vals)
            dtw_out = self.anomaly_detector.compute_dtw_pattern_deviation(test_binned, ref_curves)
            dtw_out["repetition"] = int(rep)
            dtw_results.append(dtw_out)

        self.anomaly_results["dtw_deviation"] = dtw_results
        n_shape_anomalies = sum(1 for d in dtw_results if d.get("is_shape_anomaly", False))
        self.logger.info("DTW analysis: %d/%d repetitions flagged as shape anomalies",
                         n_shape_anomalies, len(dtw_results))

    def _apply_decision_support(self) -> None:
        """Run the clinical decision-support tree and save screening results."""
        task_group, task_id = resolve_dominant_task(self.frame_data)

        has_profile = self.task_profile is not None and self.task_profile.is_loaded()
        has_ref = self.reference_session_id is not None
        ref_stats = self.reference_baseline_stats
        if has_profile and self.task_profile_ref:
            if not has_ref:
                ref_stats = self.task_profile_ref.get("per_feature_stats", ref_stats)
            else:
                profile_stats = self.task_profile_ref.get("per_feature_stats")
                if profile_stats and not ref_stats:
                    ref_stats = profile_stats

        _ref_asym_stats = None
        if hasattr(self, "reference_metrics_df") and self.reference_metrics_df is not None and len(self.reference_metrics_df) > 0:
            _asym_col = next(
                (c for c in ("mean_asymmetry_ratio", "asymmetry_ratio_mean", "mean_asymmetry")
                 if c in self.reference_metrics_df.columns),
                None,
            )
            if _asym_col:
                _ref_a = self.reference_metrics_df
                if "task_group" in self.reference_metrics_df.columns:
                    _ref_a = self.reference_metrics_df[self.reference_metrics_df["task_group"].astype(str) == "A"]
                _asym_vals = _ref_a[_asym_col].dropna().values
                if len(_asym_vals) >= 2:
                    _ref_asym_stats = {"mean": float(np.mean(_asym_vals)), "std": float(np.std(_asym_vals, ddof=1))}
                elif len(_asym_vals) == 1:
                    _ref_asym_stats = {"mean": float(_asym_vals[0]), "std": 0.05}

        _session_is_ors = (
            "ors" in self.session_label.lower()
            or "rotated" in self.session_label.lower()
            or "ors" in self.io.session_id.lower()
        )
        self.decision_support.set_session_context(
            is_baseline=self.is_baseline_session,
            has_reference=has_ref,
            reference_stats=ref_stats,
            task_group=task_group,
            task_id=task_id,
            reference_articulation=self.reference_articulation_scores,
            reference_asymmetry_stats=_ref_asym_stats,
            is_ors_session=_session_is_ors,
        )
        self.screening_results = self.decision_support.evaluate(
            self.session_metrics,
            self.task_metrics_df,
            self.repetition_metrics_df,
            self.anomaly_results,
        )
        save_json(self.screening_results, self.io.get_screening_results_path())

        confidence_summary = {
            "confidence": self.screening_results.get("confidence", {}),
            "n_indications": self.screening_results.get("n_indications", 0),
            "indication_types": self.screening_results.get("indication_types", []),
            "is_baseline_session": self.is_baseline_session,
            "reference_session": self.reference_session_id,
        }
        save_json(confidence_summary, self.io.get_confidence_summary_path())

        indications = self.screening_results.get("indications", [])
        if indications:
            trace_df = pd.DataFrame([
                {
                    "indication_type": ind.get("indication_type"),
                    "severity": ind.get("severity"),
                    "confidence": ind.get("confidence"),
                    "source_node": ind.get("source_node"),
                    "description": ind.get("description"),
                    "supporting_features": json.dumps(ind.get("supporting_features", {})),
                }
                for ind in indications
            ])
            self.io.save_dataframe(trace_df, self.io.get_decision_trace_path(), include_metadata=False)

        self.logger.info("Decision support: %d screening indications", len(indications))
        if self.is_baseline_session and indications:
            self.logger.warning("Note: Baseline session - indications should be interpreted with caution")
        for ind in indications:
            self.logger.info("  - %s (%s): confidence=%.2f", ind["indication_type"], ind["severity"], ind["confidence"])

    def _validate_baseline_quality(self) -> None:
        """Run quality checks on the baseline and warn if unacceptable."""
        if not hasattr(self.baseline_constructor, 'validate_quality'):
            return
        quality = self.baseline_constructor.validate_quality()
        if quality.get("warnings"):
            for w in quality["warnings"]:
                self.logger.warning("Baseline quality: %s", w)
        if not quality.get("is_acceptable", True):
            self.logger.warning("Baseline quality is below acceptable thresholds — results may be unreliable")
        save_json(quality, self.io.results_dir / "baseline_quality.json")
        self.session_metrics["baseline_quality"] = quality

    def _run_anatomical_analysis(self) -> None:
        """Generate anatomical muscle group analysis from anomaly results."""
        feature_devs = self.anomaly_results.get("feature_deviations", {})
        if not feature_devs:
            return
        try:
            report = generate_anatomical_report(feature_devs)
            self.anomaly_results["anatomical_report"] = report
            save_json(report, self.io.results_dir / "anatomical_analysis.json")
            self.logger.info(
                "Anatomical analysis: %d/%d muscle groups affected",
                report.get("n_groups_affected", 0),
                report.get("n_groups_total", 0),
            )
        except Exception as exc:
            self.logger.warning("Could not generate anatomical report: %s", exc)

    def _run_trend_analysis(self) -> None:
        """Run longitudinal trend analysis if multiple sessions exist."""
        try:
            subject_dir = self.io.data_dir / "results" / self.study_mode / self.subject_id
            if not subject_dir.exists():
                return

            session_summaries = []
            for session_dir in sorted(subject_dir.iterdir()):
                summary_path = session_dir / "pipeline_summary.json"
                if summary_path.exists():
                    summary = load_json(summary_path)
                    if summary:
                        flat = {"session_id": summary.get("session_id", "")}
                        flat["timestamp"] = summary.get("timestamp", "")
                        for key, val in summary.get("session_metrics", {}).items():
                            if isinstance(val, (int, float)):
                                flat[key] = val
                        anomaly_summary = summary.get("anomaly_summary", {})
                        for key, val in anomaly_summary.items():
                            if isinstance(val, (int, float)):
                                flat[f"anomaly_{key}"] = val
                        session_summaries.append(flat)

            if len(session_summaries) >= 3:
                analyzer = create_trend_analyzer()
                trends = analyzer.analyze_trends(session_summaries)
                save_json(trends, self.io.results_dir / "trend_analysis.json")
                self.session_metrics["trends"] = trends
                self.logger.info(
                    "Trend analysis: %d sessions, %d significant trends, direction: %s",
                    trends.get("n_sessions", 0),
                    trends.get("n_significant_trends", 0),
                    trends.get("overall_direction", "unknown"),
                )
        except Exception as exc:
            self.logger.warning("Could not run trend analysis: %s", exc)

    def _generate_visualizations(self) -> None:
        """Produce all PNG visualisations and PDF tables into the results directory."""
        viz_dir = self.io.results_dir / "visualizations"
        tables_dir = self.io.results_dir / "tables"

        neutral_baseline_stats = getattr(self.baseline_constructor, "baseline_stats", None)
        reference_baseline_stats = self.reference_baseline_stats if self.reference_session_id else None
        tp_ref = self.task_profile_ref
        tp_all = (
            self.task_profile.tasks
            if self.task_profile is not None and self.task_profile.is_loaded()
            else None
        )

        _is_updated_reference = (
            self.is_baseline_session
            and self.task_profile is not None
            and self.task_profile.is_loaded()
            and len(self.task_profile.sessions_included) >= 2
        )
        _show_reference_overlay = not self.is_baseline_session or _is_updated_reference

        if _is_updated_reference:
            self.logger.info(
                "New reference session with %d prior session(s) — visualizations will "
                "reflect the accumulated combined reference profile.",
                len(self.task_profile.sessions_included) - 1,
            )

        has_features = self.features_df is not None and len(self.features_df) > 0
        has_reps = self.repetition_metrics_df is not None and len(self.repetition_metrics_df) > 0
        has_anomaly_scores = bool(self.anomaly_results and self.anomaly_results.get("anomaly_scores"))

        if has_features:
            if "mean_activation" in self.features_df.columns:
                overlay_kwargs = dict(
                    baseline_stats=neutral_baseline_stats,
                    reference_baseline_stats=reference_baseline_stats,
                    task_profile_ref=tp_ref,
                    all_task_profiles=tp_all,
                )
                self.visualizer.plot_repetition_overlay(
                    self.features_df, "mean_activation",
                    viz_dir / "activation_overlay",
                    title="Activation by Repetition (Overlayed)",
                    **overlay_kwargs,
                )
                self.visualizer.plot_activation_per_repetition(
                    self.features_df, "mean_activation",
                    viz_dir / "activation_per_repetition",
                    title="Activation per Repetition",
                    **overlay_kwargs,
                )
                activation_metrics = [c for c in self.features_df.columns if "activation" in c][:4]
                if activation_metrics:
                    self.visualizer.plot_activation_overlay_by_metric(
                        self.features_df, activation_metrics,
                        viz_dir / "activation_overlay_by_metric",
                        title="Activation Overlay by Metric",
                        **overlay_kwargs,
                    )

            self.visualizer.plot_activation_overlay_by_feature_pdf(
                self.features_df,
                viz_dir / "activation_overlay_by_feature",
                title="Activation Overlay per Feature",
                baseline_stats=neutral_baseline_stats,
                reference_baseline_stats=reference_baseline_stats,
                task_profile_ref=tp_ref,
                all_task_profiles=tp_all,
            )
            self.visualizer.plot_asymmetry_over_time(
                self.features_df,
                viz_dir / "asymmetry_analysis",
                title="Facial Asymmetry Analysis",
                baseline_stats=neutral_baseline_stats,
                reference_baseline_stats=reference_baseline_stats if _show_reference_overlay else None,
                all_task_profiles=tp_all if _show_reference_overlay else None,
            )

        if has_reps:
            self.visualizer.plot_metrics_summary(
                self.repetition_metrics_df,
                viz_dir / "metrics_summary",
                title="Repetition Metrics Summary",
                baseline_stats=neutral_baseline_stats,
                reference_baseline_stats=reference_baseline_stats,
                task_profile_ref=tp_ref,
            )

        if self.screening_results:
            self.visualizer.plot_screening_summary(
                self.screening_results,
                viz_dir / "screening_summary",
                anomaly_results=self.anomaly_results,
                articulation_scores=self.articulation_scores,
                title="Clinical Screening Report",
            )
            self.visualizer.create_screening_table(
                self.screening_results,
                tables_dir / "screening_results_table",
            )
            self.visualizer.plot_disorder_evidence(
                self.screening_results,
                viz_dir / "disorder_evidence",
                title="Disorder Evidence Profile",
            )

        cross_task_path = self.io.results_dir / "cross_task_matching.json"
        if cross_task_path.exists():
            cross_task_data = load_json(cross_task_path)
            task_name_map = self.visualizer._build_task_name_map(self.repetition_metrics_df)
            self.visualizer.plot_cross_task_matching(
                cross_task_data,
                viz_dir / "cross_task_matching",
                task_name_map=task_name_map,
                title="Cross-Task Profile Matching (Group A)",
            )

        if has_anomaly_scores:
            self.visualizer.plot_anomaly_results(
                self.anomaly_results,
                viz_dir / "anomaly_results",
                title="Anomaly Detection Results",
                baseline_stats=neutral_baseline_stats,
            )
            if self.screening_results:
                try:
                    self.visualizer.plot_anomaly_indication_flow(
                        self.anomaly_results,
                        self.screening_results,
                        viz_dir / "anomaly_indication_flow",
                        title=f"Anomaly → Indication Flow",
                    )
                except Exception as _aif_exc:
                    self.logger.warning("plot_anomaly_indication_flow failed: %s", _aif_exc)
            self.visualizer.create_anomaly_table(
                self.anomaly_results,
                tables_dir / "anomaly_results_table",
            )
            task_group, task_id = resolve_dominant_task(self.frame_data)
            task_label = f"{task_group}-{task_id}" if task_group and task_group != '0' else ""
            self.visualizer.plot_deviations_summary(
                self.anomaly_results,
                self.screening_results,
                viz_dir / "deviations_summary",
                task_name=task_label,
            )

        if has_anomaly_scores:
            clinical_path = self.io.results_dir / "clinical_comparison.json"
            clinical_comparison = load_json(clinical_path) if clinical_path.exists() else None
            self.visualizer.plot_confusion_matrix(
                self.anomaly_results,
                viz_dir / "confusion_matrix",
                title="Detection Agreement Matrix",
                clinical_comparison=clinical_comparison,
            )

        if has_reps and len(self.repetition_metrics_df) >= 3:
            self.visualizer.plot_cluster_embeddings(
                self.repetition_metrics_df,
                self.anomaly_results,
                viz_dir / "cluster_embeddings",
                title="Repetition Cluster Embeddings",
                task_profile_ref=tp_ref,
            )

        if has_features:
            self.visualizer.plot_heatmap(
                self.features_df,
                viz_dir / "activation_heatmap",
                title="Blendshape Activation Heatmap",
            )
            self.visualizer.create_heatmap_table(
                self.features_df,
                tables_dir / "activation_statistics_table",
            )

            self.logger.info("Generating muscle group activation heatmap...")
            self.visualizer.plot_muscle_group_activation_heatmap(
                self.features_df,
                viz_dir / "activation_heatmap_muscle_groups",
            )

            self.logger.info("Generating muscle group temporal heatmap...")
            self.visualizer.plot_muscle_group_temporal_heatmap(
                self.features_df,
                viz_dir / "muscle_group_temporal_heatmap",
            )

        if self.study_mode == "patient" and "continuous" in self.session_metrics:
            self.visualizer.plot_intraop_timeline(
                self.session_metrics["continuous"],
                viz_dir / "intraop_timeline",
                title="Intra-operative Timeline",
            )

        if self.reference_session_id:
            ref_metrics_path = self.io.get_reference_session_path(self.reference_session_id) / "repetition_metrics.csv"
            if ref_metrics_path.exists():
                self.visualizer.plot_statistical_comparison(
                    pd.read_csv(ref_metrics_path),
                    self.repetition_metrics_df,
                    viz_dir / "baseline_comparison",
                    title=f"Comparison: Current vs Baseline ({self.reference_session_id})",
                )

        clinical_comparison_path = self.io.results_dir / "clinical_comparison.json"
        if clinical_comparison_path.exists():
            clinical_data = load_json(clinical_comparison_path)
            validation_path = self.io.results_dir / "validation_result.json"
            val_report = load_json(validation_path) if validation_path.exists() else None
            self.visualizer.plot_clinical_agreement_combined(
                clinical_data,
                val_report,
                viz_dir / "clinical_agreement",
                title="Clinical Agreement & Validation",
            )

        if self.session_metrics.get("trends"):
            self.visualizer.plot_trend_analysis(
                self.session_metrics["trends"],
                viz_dir / "trend_analysis",
                title="Longitudinal Trend Analysis",
            )

        if self.articulation_scores and self.articulation_scores.get("per_task_scores"):
            self.visualizer.plot_articulation_profile(
                self.articulation_scores,
                viz_dir / "speech_scores",
                title="Speech Scores",
                reference_scores=self.reference_articulation_scores,
            )

        if self.articulation_scores and self.articulation_scores.get("per_word_scores"):
            self.visualizer.plot_word_production_profile(
                self.articulation_scores,
                viz_dir / "word_production_profile",
                title="Word Production Profile",
                reference_scores=self.reference_articulation_scores,
            )

        if self.anomaly_results and self.anomaly_results.get("anatomical_report"):
            self.visualizer.plot_anatomical_report(
                self.anomaly_results["anatomical_report"],
                viz_dir / "anatomical_report",
                title="Anatomical Muscle Group Analysis",
            )

            ref_anat = None
            if self.reference_session_id:
                ref_anat_path = (
                    self.io.get_reference_session_path(self.reference_session_id)
                    / "anomaly_results.json"
                )
                if ref_anat_path.exists():
                    ref_anom = load_json(ref_anat_path)
                    ref_anat = ref_anom.get("anatomical_report") if ref_anom else None
            self.visualizer.plot_anatomical_comparison(
                self.anomaly_results["anatomical_report"],
                viz_dir / "anatomical_comparison",
                reference_report=ref_anat,
                title="Anatomical Muscle Group Comparison",
            )

            self.logger.info("Generating per-task anatomical comparison...")
            from src.anatomy import generate_per_repetition_anatomical_reports
            per_task_reports = generate_per_repetition_anatomical_reports(
                self.anomaly_results
            )
            if per_task_reports:
                self.visualizer.plot_anatomical_comparison_per_task(
                    per_task_reports,
                    viz_dir / "anatomical_comparison_per_task",
                    reference_report=ref_anat,
                )

        if neutral_baseline_stats:
            self.visualizer.plot_baseline_stability(
                neutral_baseline_stats,
                viz_dir / "baseline_stability",
                title="Baseline Blendshape Stability",
            )

        if self.screening_results and self.screening_results.get("indications"):
            ref_screening = None
            if self.reference_session_id:
                ref_scr_path = (
                    self.io.get_reference_session_path(self.reference_session_id)
                    / "screening_results.json"
                )
                if ref_scr_path.exists():
                    ref_screening = load_json(ref_scr_path)
            self.visualizer.create_deviation_score_table(
                self.screening_results,
                tables_dir / "deviation_score_table",
                reference_screening=ref_screening,
                title="Deviation Score Summary",
            )

        if self.session_metrics.get("trends"):
            self.visualizer.create_trend_summary_table(
                self.session_metrics["trends"],
                tables_dir / "trend_summary_table",
                title="Longitudinal Trend Summary",
            )

        fatigue_drift_path = self.io.results_dir / "fatigue_drift_report.json"
        if fatigue_drift_path.exists():
            try:
                fatigue_drift_report = load_json(fatigue_drift_path)
                if fatigue_drift_report and fatigue_drift_report.get("windows"):
                    self.visualizer.plot_fatigue_drift_report(
                        fatigue_drift_report,
                        viz_dir / "fatigue_drift_report",
                        title=f"Fatigue & Motor Drift — {self.io.session_id}",
                    )
                    self.logger.info("Generated fatigue drift visualization.")
            except Exception as _fdv_exc:
                self.logger.debug("Fatigue drift visualization skipped: %s", _fdv_exc)

        overview_csv = (
            self.io.data_dir / "processed" / self.study_mode
            / self.subject_id / f"{self.subject_id}_session_overview.csv"
        )
        if overview_csv.exists():
            overview_df = pd.read_csv(overview_csv)
            overview_records = overview_df.to_dict("records")

            self.visualizer.create_detection_quality_table(
                overview_records,
                tables_dir / "detection_quality_table",
                title="Detection Quality per Session",
            )

            if len(overview_records) >= 2:
                self.visualizer.plot_patient_trajectory(
                    overview_records,
                    viz_dir / "patient_trajectory",
                    title="Patient Trajectory",
                )

        self.logger.info("Generated visualizations in %s", viz_dir)
        self.logger.info("Generated tables in %s", tables_dir)

    def _run_validation(self, alteration_type: str) -> None:
        """Evaluate whether the pipeline's indications match the known alteration type."""
        if self.validator is None:
            return
        validation_result = self.validator.validate_session(
            self.io.session_id, alteration_type, self.screening_results
        )
        save_json(validation_result, self.io.results_dir / "validation_result.json")
        self.logger.info(
            "Validation result: %s", "CORRECT" if validation_result["is_correct"] else "MISMATCH"
        )
        self.logger.info("  Expected: %s", validation_result["expected_indications"])
        self.logger.info("  Predicted: %s", validation_result["predicted_indications"])

        subject_results_dir = self.io.data_dir / "results" / self.study_mode / self.subject_id
        all_val_results = []
        if subject_results_dir.exists():
            for sess_dir in sorted(subject_results_dir.iterdir()):
                vr_path = sess_dir / "validation_result.json"
                if vr_path.exists():
                    vr = load_json(vr_path)
                    if vr:
                        all_val_results.append(vr)

        if len(all_val_results) >= 2:
            report = self.validator.generate_validation_report(
                all_val_results,
                self.io.results_dir / "validation_report",
            )
            tables_dir = self.io.results_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self.visualizer.create_pilot_validation_table(
                report,
                tables_dir / "pilot_validation_table",
                title="Pilot Validation: Per-Indication Metrics",
            )

    def _save_results(self) -> None:
        """Persist the pipeline summary and regenerate consolidated CSVs."""
        save_json(self._get_summary(), self.io.results_dir / "pipeline_summary.json")
        self.logger.info("Results saved to %s", self.io.results_dir)
        try:
            cons_path, overview_path = consolidate_subject(
                self.io.project_root, self.subject_id, self.study_mode
            )
            if cons_path:
                self.logger.info("Consolidated CSV updated: %s", cons_path)
            if overview_path:
                self.logger.info("Session overview CSV updated: %s", overview_path)
        except Exception as exc:
            self.logger.warning("Could not generate consolidated CSVs: %s", exc)

    def _compute_continuous_metrics(self) -> None:
        """Compute windowed fatigue, asymmetry-trend, and response-latency metrics for long recordings."""
        if self.features_df is None or len(self.features_df) == 0:
            return

        continuous_metrics: Dict[str, Any] = {}
        has_activation = "mean_activation" in self.features_df.columns
        has_time = "timestamp_abs" in self.features_df.columns

        if has_activation and has_time:
            time_vals = self.features_df["timestamp_abs"].values
            activation_vals = self.features_df["mean_activation"].values
            max_time = time_vals.max()

            window_metrics = []
            for start in np.arange(0, max_time, _CONTINUOUS_WINDOW_SEC):
                mask = (time_vals >= start) & (time_vals < start + _CONTINUOUS_WINDOW_SEC)
                if mask.sum() > _MIN_CONTINUOUS_WINDOW_FRAMES:
                    w = activation_vals[mask]
                    window_metrics.append({
                        "window_start": float(start),
                        "window_end": float(start + _CONTINUOUS_WINDOW_SEC),
                        "mean_activation": float(np.mean(w)),
                        "std_activation": float(np.std(w)),
                        "max_activation": float(np.max(w)),
                    })

            if len(window_metrics) > 1:
                third = len(window_metrics) // 3
                early_mean = np.mean([w["mean_activation"] for w in window_metrics[:third]])
                late_mean = np.mean([w["mean_activation"] for w in window_metrics[-third:]])
                continuous_metrics["fatigue"] = {
                    "early_activation_mean": float(early_mean),
                    "late_activation_mean": float(late_mean),
                    "activation_decay": float(early_mean - late_mean),
                    "decay_percent": float((early_mean - late_mean) / early_mean * 100) if early_mean > 0 else 0,
                    "window_metrics": window_metrics,
                }

        asymmetry_cols = [c for c in self.features_df.columns if "asymmetry_ratio" in c]
        if asymmetry_cols and has_time:
            time_vals = self.features_df["timestamp_abs"].values
            window_asymmetry = []
            for start in np.arange(0, time_vals.max(), _CONTINUOUS_WINDOW_SEC):
                mask = (time_vals >= start) & (time_vals < start + _CONTINUOUS_WINDOW_SEC)
                if mask.sum() > _MIN_CONTINUOUS_WINDOW_FRAMES:
                    window_asymmetry.append({
                        "window_start": float(start),
                        "window_end": float(start + _CONTINUOUS_WINDOW_SEC),
                        "mean_asymmetry": float(self.features_df.loc[mask, asymmetry_cols].abs().mean().mean()),
                    })

            if len(window_asymmetry) > 1:
                times = np.array([w["window_start"] for w in window_asymmetry])
                asyms = np.array([w["mean_asymmetry"] for w in window_asymmetry])
                slope = np.polyfit(times, asyms, 1)[0]
                third = len(asyms) // 3
                continuous_metrics["asymmetry_trend"] = {
                    "slope_per_minute": float(slope * 60),
                    "is_increasing": bool(slope > 0.001),
                    "early_asymmetry": float(np.mean(asyms[:third])),
                    "late_asymmetry": float(np.mean(asyms[-third:])),
                    "window_data": window_asymmetry,
                }

        if self.events_df is not None and len(self.events_df) > 0:
            measurement_events = self.events_df[self.events_df["event_type"] == "measurement"]
            if len(measurement_events) > 2:
                latencies = measurement_events["timestamp_abs"].diff().dropna().values
                continuous_metrics["response_latency"] = {
                    "mean_latency": float(np.mean(latencies)),
                    "std_latency": float(np.std(latencies)),
                    "trend": float(np.polyfit(range(len(latencies)), latencies, 1)[0]) if len(latencies) > 2 else 0,
                }

        if continuous_metrics:
            save_json(continuous_metrics, self.io.results_dir / "continuous_metrics.json")
            self.session_metrics["continuous"] = continuous_metrics
            self.logger.info("Computed continuous session metrics: %s", list(continuous_metrics.keys()))

        try:
            from .anomaly import FatigueDriftMonitor

            if (
                self.features_df is not None
                and has_time
                and (self.features_df["timestamp_abs"].max() - self.features_df["timestamp_abs"].min()) >= 60.0
            ):
                fatigue_monitor = FatigueDriftMonitor(
                    baseline_duration_s=120.0,
                    window_size_s=60.0,
                    step_size_s=10.0,
                ).fit(self.features_df)

                fatigue_report = fatigue_monitor.analyze(self.features_df)
                save_json(fatigue_report, self.io.results_dir / "fatigue_drift_report.json")
                continuous_metrics["fatigue_drift"] = fatigue_report["summary"]
                if continuous_metrics:
                    save_json(continuous_metrics, self.io.results_dir / "continuous_metrics.json")
                self.logger.info(
                    "Fatigue drift monitor: %d windows, %d flagged (%.0f%% of session).",
                    fatigue_report["summary"]["n_windows"],
                    fatigue_report["summary"]["n_flagged"],
                    fatigue_report["summary"]["flag_fraction"] * 100,
                )
        except Exception as _fdm_exc:
            self.logger.debug("Fatigue drift monitor skipped: %s", _fdm_exc)

    def _compare_clinical_notes(self, clinical_notes: Dict[str, Any]) -> None:
        """Cross-reference ML screening indications against clinician-provided observations."""
        clinical_indications = set(clinical_notes.get("observed_conditions", []))
        ml_indications = {ind["indication_type"] for ind in self.screening_results.get("indications", [])}

        matches = clinical_indications & ml_indications
        clinical_only = clinical_indications - ml_indications
        ml_only = ml_indications - clinical_indications

        discrepancies = [
            {"type": "missed_by_ml", "condition": c, "note": f"Clinical observation '{c}' not detected by ML pipeline"}
            for c in clinical_only
        ] + [
            {"type": "ml_prediction", "condition": c, "note": f"ML predicted '{c}' - not in clinical notes (may be subclinical)"}
            for c in ml_only
        ]

        comparison = {
            "clinical_observations": clinical_notes,
            "ml_predictions": {
                "indications": self.screening_results.get("indications", []),
                "confidence": self.screening_results.get("confidence", {}),
                "anomalies": self.anomaly_results.get("summary", {}),
            },
            "agreement": {
                "matches": sorted(matches),
                "clinical_only": sorted(clinical_only),
                "ml_only": sorted(ml_only),
                "agreement_rate": len(matches) / max(len(clinical_indications), 1),
            },
            "discrepancies": discrepancies,
        }
        save_json(comparison, self.io.results_dir / "clinical_comparison.json")
        self.logger.info(
            "Clinical comparison: %d matches, %d missed, %d ML-only",
            len(matches), len(clinical_only), len(ml_only),
        )

    def _get_summary(self) -> Dict[str, Any]:
        """Build a JSON-serialisable dict summarising the pipeline run."""
        tasks_performed: List[str] = []
        reps_per_task: Dict[str, Dict] = {}
        total_reps = 0

        if self.repetition_metrics_df is not None and len(self.repetition_metrics_df) > 0:
            for _, row in self.repetition_metrics_df.iterrows():
                tg = row.get("task_group", "0")
                tid = row.get("task_id", 0)
                task_name = row.get("task_name", "")
                if tg and str(tg) not in ("0", "nan"):
                    task_key = f"{tg}{tid}" if tid and tid != 0 else str(tg)
                    task_label = task_name if task_name and task_name != "(no task selected)" else task_key
                    if task_key not in reps_per_task:
                        reps_per_task[task_key] = {"label": task_label, "count": 0}
                        tasks_performed.append(task_label)
                    reps_per_task[task_key]["count"] += 1
                total_reps += 1

        return {
            "session_id": self.io.session_id,
            "subject_id": self.subject_id,
            "session_label": self.session_label,
            "study_mode": self.study_mode,
            "pipeline_version": get_pipeline_version(),
            "config_hash": self.io.config_hash,
            "timestamp": datetime.now().isoformat(),
            "n_frames": len(self.frame_data),
            "n_tasks": len(tasks_performed),
            "tasks_performed": tasks_performed,
            "reps_per_task": reps_per_task,
            "n_repetitions": total_reps,
            "session_metrics": self.session_metrics,
            "screening_summary": {
                "n_indications": self.screening_results.get("n_indications", 0),
                "indication_types": self.screening_results.get("indication_types", []),
                "confidence": self.screening_results.get("confidence", {}),
            },
            "anomaly_summary": self.anomaly_results.get("summary", {}),
            "articulation_summary": (
                {
                    "mean_score": self.articulation_scores.get("mean_articulation_score"),
                    "pataka_score": self.articulation_scores.get("articulation_score_pataka"),
                    "simple_syllable_mean": self.articulation_scores.get("simple_syllable_mean"),
                    "impairment_consistency": self.articulation_scores.get("articulation_impairment_consistency"),
                    "n_tasks_scored": self.articulation_scores.get("n_tasks_scored", 0),
                    "word_production_quality": self.articulation_scores.get("word_production_quality"),
                    "complexity_gradient": self.articulation_scores.get("complexity_gradient"),
                    "n_words_scored": self.articulation_scores.get("n_words_scored", 0),
                    "has_reference": self.reference_articulation_scores is not None,
                    "reference_mean": (
                        self.reference_articulation_scores.get("mean_articulation_score")
                        if self.reference_articulation_scores else None
                    ),
                    "delta_mean": (
                        self.articulation_scores.get("mean_articulation_score", 0)
                        - self.reference_articulation_scores.get("mean_articulation_score", 0)
                        if self.reference_articulation_scores
                        and self.articulation_scores.get("mean_articulation_score") is not None
                        else None
                    ),
                }
                if self.articulation_scores
                else None
            ),
            "output_paths": {
                "raw": str(self.io.raw_dir),
                "processed": str(self.io.processed_dir),
                "results": str(self.io.results_dir),
            },
        }


def _build_normative_command(search_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Build a normative reference from all pipeline_summary.json files found
    under search_dir and write normative_reference.json to output_dir
    (defaults to search_dir if not specified).

    Returns a dict with n_sessions and n_features for caller reporting.
    """
    from .baseline import build_normative_reference, save_normative_reference
    from .utils import load_json

    search_dir = Path(search_dir)
    out_dir = Path(output_dir) if output_dir is not None else search_dir
    summaries = []
    for p in search_dir.rglob("pipeline_summary.json"):
        try:
            data = load_json(p)
            metrics = data.get("session_metrics", data)
            if metrics and any(isinstance(v, (int, float)) for v in metrics.values()):
                summaries.append({"metrics": metrics})
        except Exception:
            pass
    if not summaries:
        print(f"No pipeline_summary.json files found under {search_dir}.")
        return {"n_sessions": 0, "n_features": 0}
    ref = build_normative_reference(summaries)
    out = out_dir / "normative_reference.json"
    save_normative_reference(ref, out)
    print(f"Normative reference built from {len(summaries)} sessions → {out}")
    print(f"  Features: {len(ref)}")
    return {"n_sessions": len(summaries), "n_features": len(ref)}


def main():
    """CLI entry point — parse arguments, instantiate and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Facial Motor and Speech Behavior Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Live capture (pilot study):
    python run_pipeline.py --mode pilot --subject P001 --session baseline --input live

  Video file (patient study):
    python run_pipeline.py --mode patient --subject PAT001 --session pre_op --input video.mp4

  With reference comparison:
    python run_pipeline.py --mode patient --subject PAT001 --session intra_op --input live --reference P001_baseline_20260101_120000

  Pilot validation with known alteration:
    python run_pipeline.py --mode pilot --subject P002 --session altered1 --input video.mp4 --alteration deliberate_asymmetry

  Annotated video (offline analysis with manual timings):
    python run_pipeline.py --mode patient --subject PAT001 --session pre_op --video recording.mp4 --annotations recording.json

  Auto-detect segments from motion:
    python run_pipeline.py --mode pilot --subject P001 --session test --video recording.mp4 --auto-detect

  Continuous non-task-based recording:
    python run_pipeline.py --mode patient --subject PAT001 --session intra_op --video recording.mp4 --continuous

  Study Prompter recordings (single profile):
    python run_pipeline.py --mode pilot --subject P001 --session test1 \\
      --prompter-videos P001_cam1_*.mp4 \\
      --prompter-timestamps P001_timestamps_*.csv \\
      --prompter-meta P001_recording_meta_*.json

  List all sessions for a subject:
    python run_pipeline.py --list-sessions --mode pilot --subject P001

  List all sessions across all subjects and modes:
    python run_pipeline.py --list-sessions

  Study Prompter with auto-discovered reference (no --reference needed):
    python run_pipeline.py --mode pilot --subject P001 --session test2 \\
      --prompter-videos P001_cam1.mp4 \\
      --prompter-timestamps P001_timestamps.csv

  Study Prompter with explicit reference sessions:
    python run_pipeline.py --mode pilot --subject P001 --session test1 \\
      --prompter-videos P001_cam1_*.mp4 \\
      --prompter-timestamps P001_timestamps_*.csv \\
      --reference P001_baseline_20260101_120000
        """,
    )
    parser.add_argument("--mode", "-m", required=False, choices=["pilot", "patient"],
                        help="Study mode: pilot (healthy participants) or patient (clinical)")
    parser.add_argument("--subject", "-s", required=False,
                        help="Subject ID (e.g., P001, PAT001)")
    parser.add_argument("--session", "-l", required=False,
                        help="Session label (e.g., baseline, pre_op, intra_op)")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List all recorded sessions (optionally filtered by --mode/--subject) and exit")
    parser.add_argument("--input", "-i", default=None,
                        help='Input source: "live" for camera or path to video file')
    parser.add_argument("--reference", "-r", nargs="*", default=None,
                        help="Reference session ID(s) for comparison (optional; comma-separated or repeated: --reference REF1 REF2)")
    parser.add_argument("--alteration", "-a", default=None,
                        help="Alteration type for pilot study validation (optional)")
    parser.add_argument("--task", "-t", default=None,
                        help="Task identifier for video processing (e.g., A2 for smiling, B1 for Pa-Pa-Pa)")
    parser.add_argument("--start-time", default=None,
                        help="Start timestamp for video processing (format: MM:SS or HH:MM:SS)")
    parser.add_argument("--end-time", default=None,
                        help="End timestamp for video processing (format: MM:SS or HH:MM:SS)")
    parser.add_argument("--clinical-notes", "-c", default=None,
                        help="Path to clinical notes JSON file for comparison with ML predictions")
    parser.add_argument("--reprocess-session", default=None,
                        help="Session ID to reprocess (delete processed/results and re-run using existing session folder)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to pre-recorded video file for offline processing")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Path to JSON annotation file with task timings")
    parser.add_argument("--auto-detect", action="store_true",
                        help="Auto-detect task segments from motion (saves annotations for review)")
    parser.add_argument("--continuous", action="store_true",
                        help="Treat video as continuous (non-task-based) recording")
    parser.add_argument(
        "--normative-reference", type=Path, default=None,
        help="Path to normative_reference.json built from healthy participant sessions; enables normative comparison in anomaly detection.",
    )
    parser.add_argument(
        "--build-normative", type=Path, default=None, metavar="SUBJECTS_DIR",
        help="Build a normative reference from all session_summary.json files found under SUBJECTS_DIR; no analysis is run; program exits after building.",
    )

    prompter_group = parser.add_argument_group("Study Prompter Input")
    prompter_group.add_argument(
        "--prompter-videos", nargs="+", default=None,
        help="1-4 camera video files from study-prompter (e.g., P001_cam1_*.mp4 P001_cam2_*.mp4)",
    )
    prompter_group.add_argument(
        "--prompter-timestamps", default=None,
        help="Timestamps CSV from study-prompter (e.g., P001_timestamps_*.csv)",
    )
    prompter_group.add_argument(
        "--prompter-assembly", default=None,
        help="Assembly CSV from study-prompter COMBINED profile (optional)",
    )
    prompter_group.add_argument(
        "--prompter-meta", default=None,
        help="Recording metadata JSON from study-prompter (optional, for start-offset)",
    )

    args = parser.parse_args()

    if args.list_sessions:
        io_root = PROJECT_ROOT
        modes = [args.mode] if args.mode else ["pilot", "patient"]
        subjects_filter = [args.subject] if args.subject else None

        any_printed = False
        for mode in modes:
            raw_mode_dir = io_root / "data" / "raw" / mode
            if not raw_mode_dir.exists():
                continue
            subject_dirs = (
                [raw_mode_dir / s for s in subjects_filter]
                if subjects_filter
                else sorted(raw_mode_dir.iterdir())
            )
            for subj_dir in subject_dirs:
                if not subj_dir.is_dir():
                    continue
                subject_id = subj_dir.name
                try:
                    tmp_io = IOManager(io_root, subject_id, "_list", mode, list_only=True)
                    sessions = tmp_io.list_sessions_with_metadata(subject_id, mode)
                except Exception:
                    sessions = []
                if not sessions:
                    continue
                print(f"\n{'='*60}")
                print(f"Subject: {subject_id}  [mode: {mode}]")
                print(f"{'='*60}")
                col_w = {"session_id": 40, "label": 16, "created_at": 22, "n_ind": 6, "ref": 5}
                header = (
                    f"{'Session ID':<{col_w['session_id']}}  "
                    f"{'Label':<{col_w['label']}}  "
                    f"{'Created':<{col_w['created_at']}}  "
                    f"{'#Ind':>{col_w['n_ind']}}  "
                    f"{'Ref?':>{col_w['ref']}}"
                )
                print(header)
                print("-" * len(header))
                for s in sessions:
                    has_ref = "yes" if s.get("has_reference_data") else "no"
                    print(
                        f"{s.get('session_id', ''):<{col_w['session_id']}}  "
                        f"{s.get('session_label', ''):<{col_w['label']}}  "
                        f"{s.get('created_at', ''):<{col_w['created_at']}}  "
                        f"{s.get('n_indications', 0):>{col_w['n_ind']}}  "
                        f"{has_ref:>{col_w['ref']}}"
                    )
                any_printed = True
        if not any_printed:
            print("No sessions found.")
        return 0

    if args.build_normative:
        _build_normative_command(Path(args.build_normative))
        return 0

    if not args.mode:
        parser.error("--mode is required")
    if not args.subject:
        parser.error("--subject is required")
    if not args.session:
        parser.error("--session is required")

    if not args.input and not args.video and not args.prompter_videos and not args.prompter_timestamps:
        parser.error("Either --input, --video, or --prompter-videos/--prompter-timestamps is required")

    if args.prompter_videos or args.prompter_timestamps:
        if not args.prompter_timestamps:
            parser.error("--prompter-timestamps is required when using --prompter-videos")
        if not args.prompter_videos:
            parser.error("--prompter-videos is required when using --prompter-timestamps")

        from src.prompter_pipeline import run_prompter_session

        reference_ids_prompter = None
        if args.reference:
            reference_ids_prompter = []
            for ref in args.reference:
                if "," in ref:
                    reference_ids_prompter.extend(
                        [r.strip() for r in ref.split(",") if r.strip()]
                    )
                else:
                    reference_ids_prompter.append(ref)
            reference_ids_prompter = reference_ids_prompter or None

        video_paths = [Path(v) for v in args.prompter_videos]
        summary = run_prompter_session(
            video_paths=video_paths,
            timestamps_path=Path(args.prompter_timestamps),
            subject_id=args.subject,
            session_label=args.session,
            study_mode=args.mode,
            project_root=PROJECT_ROOT,
            meta_path=Path(args.prompter_meta) if args.prompter_meta else None,
            assembly_path=Path(args.prompter_assembly) if args.prompter_assembly else None,
            reference_session=reference_ids_prompter,
        )

        print("\n" + "=" * 60)
        print("STUDY PROMPTER PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Participant: {summary['participant_id']}")
        print(f"Profile:     {summary['profile']}")
        print(f"Date:        {summary['session_date']}")
        print(f"COMBINED:    {summary['is_combined']}")
        for disorder_result in summary.get("disorder_results", []):
            disorder_key = disorder_result.get("disorder_key", "")
            label = f"  [{disorder_key}] " if disorder_key else "  "
            scr = disorder_result.get("screening_summary", {})
            print(
                f"{label}Session: {disorder_result.get('session_id', '')} — "
                f"{scr.get('n_indications', 0)} indication(s) "
                f"[confidence: {scr.get('confidence', {}).get('overall', 0):.2f}]"
            )
        print(f"\nOutput root: {summary['output_root']}")
        print("=" * 60 + "\n")
        return 0

    if not args.input and not args.video:
        parser.error("Either --input or --video is required")

    if args.video:
        video_path = Path(args.video)
        annotation_path = Path(args.annotations) if args.annotations else None

        process_video_file(
            video_path=video_path,
            annotation_path=annotation_path,
            subject_id=args.subject,
            session_label=args.session,
            study_mode=args.mode,
            auto_detect=args.auto_detect,
            continuous=args.continuous,
        )
        return 0

    task_info = None
    if args.input and args.input.lower() != "live":
        task_info = args.task if args.task else _prompt_for_task()

    start_time_sec = _parse_timestamp(args.start_time) if args.start_time else None
    end_time_sec = _parse_timestamp(args.end_time) if args.end_time else None
    clinical_notes = _load_clinical_notes(args.clinical_notes) if args.clinical_notes else None

    reference_ids = None
    if args.reference:
        reference_ids = []
        for ref in args.reference:
            if "," in ref:
                reference_ids.extend([r.strip() for r in ref.split(",") if r.strip()])
            else:
                reference_ids.append(ref)
        reference_ids = reference_ids if reference_ids else None

    print("\n" + "=" * 60)
    print("FACIAL MOTOR AND SPEECH BEHAVIOR ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Version: {get_pipeline_version()}")
    print(f"Mode: {args.mode}")
    print(f"Subject: {args.subject}")
    print(f"Session: {args.session}")
    print(f"Input: {args.input}")
    if task_info:
        print(f"Task: {task_info}")
    if start_time_sec is not None or end_time_sec is not None:
        print(f"Time range: {args.start_time or 'start'} to {args.end_time or 'end'}")
    if clinical_notes:
        print("Clinical notes: loaded")
    print("=" * 60 + "\n")

    pipeline = Pipeline(study_mode=args.mode, subject_id=args.subject, session_label=args.session)

    if args.reprocess_session:
        print(f"Reprocessing session: {args.reprocess_session} — clearing processed/results and re-running")
        pipeline.io.set_session_id(args.reprocess_session)
        pipeline.io.delete_processed_and_results(args.reprocess_session)

    summary = pipeline.run(
        input_source=args.input,
        reference_session=reference_ids,
        alteration_type=args.alteration,
        task_info=task_info,
        start_time=start_time_sec,
        end_time=end_time_sec,
        clinical_notes=clinical_notes,
    )

    try:
        _tools_dir = str(PROJECT_ROOT / "tools")
        if _tools_dir not in sys.path:
            sys.path.insert(0, _tools_dir)
        from session_summary_figure import generate_session_summary as _gen_summary, generate_participant_summary as _gen_participant
        _results_dir = Path(summary["output_paths"]["results"])
        _gen_summary(_results_dir)
        _subject_results_dir = _results_dir.parent if any(
            summary.get("session_id", "").count("_") > 2
        ) else _results_dir
        _participant_results = PROJECT_ROOT / "data" / "results" / summary.get("study_mode", "pilot") / summary.get("subject_id", "")
        if _participant_results.exists():
            _gen_participant(_participant_results)
    except Exception as _exc:
        pass

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Session ID: {summary['session_id']}")
    print(f"Frames processed: {summary['n_frames']}")

    n_tasks = summary.get("n_tasks", 0)
    if n_tasks > 0:
        print(f"Tasks performed: {n_tasks}")
        for _key, info in summary.get("reps_per_task", {}).items():
            print(f"  - {info['label']}: {info['count']} repetitions")
        print(f"Total repetitions: {summary['n_repetitions']}")
    else:
        print("Tasks performed: (no task selected)")
        print(f"Repetitions: {summary['n_repetitions']}")

    print(f"Screening indications: {summary['screening_summary']['n_indications']}")
    if summary["screening_summary"]["indication_types"]:
        print(f"Indication types: {', '.join(summary['screening_summary']['indication_types'])}")
    print(f"Overall confidence: {summary['screening_summary']['confidence'].get('overall', 0):.2f}")
    print(f"\nResults saved to: {summary['output_paths']['results']}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
