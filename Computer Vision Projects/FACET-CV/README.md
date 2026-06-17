# FACET-CV: Facial Analysis for Computational Evaluation and Tracking through Computer Vision

A reproducible, camera-only pipeline for quantifying facial motor control and speech production from video recordings of a standardised task battery. Designed for awake craniotomy research: the same pipeline processes a normative pilot cohort and pre-/intra-/post-operative patient recordings, enabling group-level characterisation and within-patient longitudinal comparison.

> **Non-diagnostic.** All outputs are for my research during my Master Thesis Neuroscience at EUR/EMC/Brain Echo Lab.
> **Author.** Cindy Steward, 2026

---

## Table of Contents

1. [Research Context](#1-research-context)
2. [Study Design](#2-study-design)
3. [Disorder Profiles](#3-disorder-profiles)
4. [Task Battery](#4-task-battery)
5. [File Structure](#5-file-structure)
6. [Installation](#6-installation)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Browser UI](#8-browser-ui)
9. [Output Files](#9-output-files)
10. [Methodology](#10-methodology)
11. [Disorder Screening Logic](#11-disorder-screening-logic)
12. [Cross-Participant Analysis](#12-cross-participant-analysis)
13. [Subject Consolidation](#13-subject-consolidation)
14. [Configuration](#14-configuration)
15. [Dependencies](#15-dependencies)
16. [Key References](#16-key-references)

---

## 1. Research Context

Awake craniotomy with direct cortical stimulation is the standard of care for tumours near eloquent cortex. Real-time monitoring of motor and language function during resection determines safe margins [Kanno & Mikuni 2015; Collée et al. 2023]. Current intraoperative monitoring relies on subjective clinical observation: a surgeon or speech-language pathologist watches the patient perform tasks and judges whether function is preserved. This is not quantified, is hard to reproduce across sites, and leaves no objective record.

Automated video-based facial motor assessment has been demonstrated as feasible in several clinical and near-clinical settings [Frajtag et al. 2025; Heinrich et al. 2025], and webcam-based speech kinematics have been analytically validated against electromagnetic articulography [Simmatis et al. 2023]. This pipeline operationalises both into a single workflow.

The pipeline delivers:
- A quantitative, session-level screening result across five disorder profiles
- Longitudinal within-patient comparison across Pre, Intra, and Post timepoints
- A normative pilot cohort baseline against which patient scores are contextualised
- Per-session academic figures and a pilot-study overview PDF suitable for publication

---

## 2. Study Design

| Mode | Population | Baseline source | Timepoints |
|------|-----------|----------------|------------|
| `pilot` | Healthy volunteers | Intra-subject baseline session (upright and/or simulated-OR position) | Baseline, Test |
| `patient` | Surgical patients | Patient's own pre-operative session | Pre, Intra, Post |

**Pilot sessions** characterise the pipeline's sensitivity and specificity under controlled conditions. Participants perform both a normal (baseline) session and a COMBINED session in which they intentionally simulate each disorder profile in sequence. This provides ground-truth-labelled data across all five disorder types.

**Patient sessions** apply the same analysis to genuine clinical recordings. The pipeline compares each session against the same patient's most recent pre-operative session and flags deviations that exceed threshold.

---

## 3. Disorder Profiles

Five neurologically grounded disorder profiles are screened simultaneously. Each corresponds to a distinct neural substrate and produces a characteristic kinematic signature that the pipeline targets.

| Code | Name | Core kinematic signature | Key neural substrate |
|------|------|--------------------------|---------------------|
| **P1** | Facial paresis | Persistent L-R amplitude asymmetry across Group A tasks; eyelid aperture asymmetry | Contralateral M1 (lower face) or CN VII pathway [Heinrich et al. 2025] |
| **P2** | Buccofacial apraxia | Task substitution errors; low cross-task profile similarity; execution-correctness deficit with preserved amplitude | Lateral premotor cortex; Broca's area (BA44) [Collée et al. 2023] |
| **P3** | Dysarthria | DDK rate and regularity reduction; amplitude decay; temporal slowdown across Groups B and A | Corticobulbar tract; cerebellum; brainstem |
| **P4** | Speech apraxia | Disproportionate decline on pa-ta-ka (B4) vs simple syllables; excess jaw effort on B4; temporal initiation delay | Premotor cortex; SMA; insula [Collée et al. 2023] |
| **P5** | Phonological disorder | Consistent word-production errors in Group C with intact Group B DDK articulation; C-task DTW elevation | Wernicke's area; supramarginal gyrus; angular gyrus [Collée et al. 2022] |

Mixed presentations (multiple positive profiles simultaneously) are fully supported.

---

## 4. Task Battery

Defined in `config/tasks.yaml`. All tasks are performed in a fixed order, guided by the study prompter tool.

### Group A: Non-speech facial motor tasks (3 repetitions each)

| ID | Task | Primary blendshapes |
|----|------|---------------------|
| A_1 | Pursing lips | mouthPucker, mouthFunnel |
| A_2 | Broad smile | mouthSmileLeft/Right |
| A_3 | Showing teeth | mouthUpperUpLeft/Right |
| A_4 | Tongue protrusion | jawOpen, mouthClose |
| A_5 | Tongue to corner | mouthLeft/Right |
| A_6 | Tongue to upper lip | mouthShrugUpper |
| A_7 | Frowning | browDownLeft/Right |
| A_8 | Puffing cheeks | cheekPuff |
| A_9 | Raising eyebrows | browOuterUpLeft/Right |

### Group B: Diadochokinetic (DDK) syllable tasks

| ID | Task | Clinical metric |
|----|------|-----------------|
| B_1 | pa-pa-pa | DDK rate, STI, D_mean |
| B_2 | ta-ta-ta | DDK rate, STI, D_mean |
| B_3 | ka-ka-ka | DDK rate, STI, D_mean |
| B_4 | pa-ta-ka | Complexity gradient; apraxia gate |

DDK clinical metrics (rate, STI, D_mean, Tsd, speed percentiles) follow Allison et al. [2022] and Simmatis et al. [2023].

### Group C: Word production tasks (2 repetitions, complexity 1-8)

Eight words with increasing phonological complexity. Group C targets phonological disorder (P5) and speech apraxia (P4) via DTW-based pattern analysis and kinematic amplitude deviation on complex words.

### COMBINED sessions: disorder-simulation tasks

In pilot COMBINED sessions, participants intentionally simulate disorder profiles. Group A simulation tasks (IDs A_10-A_17) are unilateral/asymmetric versions of standard tasks, resolved back to bilateral references at anomaly-detection time:

| Disorder task | Simulates | Reference |
|--------------|-----------|-----------|
| A_10 Lip purse (lateral) | Facial paresis | A_1 |
| A_11 Smile (one side) | Facial paresis | A_3 |
| A_12 Broad smile (one side) | Facial paresis | A_3 |
| A_13 Tongue lateral (left) | Facial paresis | A_5 |
| A_14 Tongue lateral (right) | Facial paresis | A_5 |
| A_15 Frown (one brow) | Facial paresis | A_7 |
| A_16 Puff right cheek | Facial paresis | A_8 |
| A_17 Raise one brow | Facial paresis | A_9 |

Group B disorder tasks are normalised to canonical IDs 1-4 by label matching. Group C disorder tasks are renumbered to canonical IDs 1-8 by sequential position. The anomaly detector receives no hint about what deviations to expect.

---

## 5. File Structure

```
master_project/
|-- config/
|   |-- decision_rules.yaml      # Screening thresholds and disorder rules
|   |-- features.yaml            # Blendshape selection, fatigue norms, normalisation
|   |-- plotting.yaml            # Figure style, Wong palette, DPI
|   `-- tasks.yaml               # Task groups A/B/C: blendshapes, symmetry pairs, reps
|
|-- data/
|   |-- raw/                     # Immutable inputs (video, CSV, JSON from study prompter)
|   |   |-- pilot/{subject}/{session_id}/
|   |   `-- patient/{subject}/{session_id}/
|   |-- processed/               # Per-frame features; repetition and task metrics
|   |   |-- pilot/{subject}/{session_id}/
|   |   `-- patient/{subject}/{session_id}/
|   `-- results/                 # All analysis outputs and figures
|       |-- pilot/{subject}/{session_id}/
|       |-- pilot/{subject}/{combined_id}/
|       |   `-- {disorder_key}/
|       `-- patient/{subject}/{session_id}/
|
|-- models/
|   `-- face_landmarker.task     # MediaPipe FaceLandmarker (downloaded by setup.py)
|
|-- src/
|   |-- anatomy.py               # Muscle group map, neural substrate mapping
|   |-- anomaly.py               # Deviation scoring, OC-SVM/IForest, DTW, fatigue drift monitoring
|   |-- articulation.py          # Per-task articulation scoring (SPARC for DDK, LDJ for words)
|   |-- baseline.py              # Baseline construction and z-score normalisation
|   |-- brain.py                 # Brain region activation map renderer
|   |-- capture.py               # Camera enumeration and live-capture wrapper
|   |-- consolidate.py           # Cross-session subject-level consolidation
|   |-- cross_participant.py     # Group-level analysis and figures
|   |-- decision_support.py      # Rule-engine screening logic, disorder confidence scoring
|   |-- feature_extraction.py    # Per-frame blendshape + landmark + head-pose extraction
|   |-- head_pose.py             # Yaw/pitch/roll from landmarks; supine correction
|   |-- io_manager.py            # All path management
|   |-- kinematic_speech.py      # Group A/B/C kinematic features; DDK clinical metrics
|   |-- metrics.py               # Repetition/task/session aggregation
|   |-- multi_camera_processor.py # Multi-camera sync, best-camera selection
|   |-- preprocessing.py         # Frame quality filtering, brightness normalisation
|   |-- prompter_pipeline.py     # Orchestrator for study-prompter COMBINED sessions
|   |-- rescreen.py              # Utility for re-running screening on saved results
|   |-- run_pipeline.py          # CLI entry point (live camera + pre-recorded video)
|   |-- study_prompter_reader.py # Parses timestamps.csv + recording_meta.json + assembly.csv
|   |-- task_profile.py          # Cross-task profile matching for buccofacial apraxia
|   |-- trends.py                # Longitudinal trend analysis
|   |-- utils.py                 # Logging, YAML/JSON helpers
|   |-- validation.py            # Input validation and data-quality assertions
|   |-- video_processor.py       # Frame-level MediaPipe inference
|   `-- visualization.py         # All figure generation (PDF/PNG)
|
|-- templates/
|   `-- index.html               # Flask upload UI
|
|-- tools/
|   `-- session_summary_figure.py    # Standalone academic 7-panel PDF generator
|
|-- launch_ui.py                 # Starts Flask browser UI
|-- requirements.txt
|-- setup.py                     # Creates venv, installs dependencies, downloads model
```

---

## 6. Installation

Requires **Python 3.11**.

```bash
cd master_project
python3 setup.py          # creates venv/, installs dependencies, downloads MediaPipe model
source venv/bin/activate
```

To install manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 7. Running the Pipeline

All commands run from `master_project/` with the venv active.

### Live camera

```bash
python src/run_pipeline.py --mode pilot --subject P001 --session baseline --input live
```

### Pre-recorded video

```bash
python src/run_pipeline.py --mode pilot --subject P001 --session test \
    --input /path/to/recording.mp4
```

### Study-prompter session (single disorder profile)

```bash
python src/run_pipeline.py --mode pilot --subject PAC1 --session test \
    --prompter-videos rec_cam0.mp4 \
    --prompter-timestamps timestamps.csv \
    --prompter-meta recording_meta.json
```

### Study-prompter session (COMBINED: all five profiles in one recording)

```bash
python src/run_pipeline.py --mode pilot --subject PAC1 --session combined \
    --prompter-videos cam0.mp4 cam1.mp4 \
    --prompter-timestamps timestamps.csv \
    --prompter-assembly assembly.csv \
    --prompter-meta recording_meta.json
```

Output: one subfolder per disorder profile under `data/results/pilot/PAC1/{combined_id}/{disorder_key}/`.

### With a reference session (baseline comparison)

```bash
python src/run_pipeline.py --mode patient --subject PAT001 --session intra_op \
    --input live \
    --reference PAT001_pre_op_20260101_090000
```

Multiple references are averaged:

```bash
python src/run_pipeline.py --mode patient --subject PAT001 --session post_op \
    --input live \
    --reference PAT001_pre_op_20260101_090000 PAT001_baseline_20260101_120000
```

**Automatic reference discovery:** When no `--reference` flag is given, `_discover_reference_session` in `src/prompter_pipeline.py` scans the subject's raw data directory for sessions with `baseline` or `normal` in the session label. It prefers a baseline that matches the current session's condition (e.g. upright vs ORS/supine), falling back to the most recently created baseline when no condition match exists.


### Regenerating specific figures

```bash
python tools/session_summary_figure.py data/results/pilot/PAC1/PAC1_test_upright_20260101_121212
```

### Subject-level consolidation

```bash
python -m src.consolidate --subject PAC1 --mode pilot
```

Output: `data/results/pilot/PAC1/{subject}_detection_quality_summary.pdf` and `{subject}_condition_comparison.pdf`.

### Cross-participant group analysis

```bash
python -m src.cross_participant \
    --subjects PAC1 PAC2 PAC3 PAC1 \
    --mode pilot \
    --demographics demographics.csv \
    --output_dir data/results/group
```

Or via Python:

```python
from src.cross_participant import compare_participants
from pathlib import Path

results = compare_participants(
    project_root=Path("master_project"),
    subject_ids=["PAC1", "PAC2", "PAC3", "PAC1"],
    study_mode="pilot",
    output_dir=Path("data/results/group"),
)
```

Output: `group_session_overview.csv`, `group_aggregated.csv`, `group_boxplots_overview.pdf`, `group_boxplots_deviation.pdf`, `group_correlation_matrix.pdf`.

---

## 8. Browser UI

```bash
python launch_ui.py --port 5050
```

Open `http://localhost:5050`. Upload the study-prompter output files (timestamps CSV, recording meta JSON, optionally assembly CSV for COMBINED sessions). The UI runs the full pipeline and streams log output in real time. Raw landmark export (CSV only, no analysis) is available via the dedicated button.

Multiple sessions can be run concurrently by opening the app in separate browser tabs. Each tab receives an independent job ID and tracks its own progress. Because MediaPipe inference is CPU-bound, throughput is best when the number of concurrent sessions does not exceed the number of available CPU cores.

---

## 9. Output Files

All outputs are written to `data/results/{mode}/{subject}/{session_id}/`.
For COMBINED sessions: `data/results/{mode}/{subject}/{combined_id}/{disorder_key}/`.

### JSON outputs

| File | Contents |
|------|----------|
| `pipeline_summary.json` | Top-level summary: session_id, subject, mode, screening_summary, anomaly_summary |
| `session_metrics.json` | overall_mean_asymmetry, overall_detection_rate, total_duration_sec, dominant_side |
| `screening_results.json` | Disorder indications: indication_type, severity, confidence, supporting_features, top_features |
| `articulation_scores.json` | mean_articulation_score, group_b_articulation_score, group_c_articulation_score, per_task_scores, per_task_deviations |
| `anomaly_results.json` | Per-repetition deviation_score, is_anomaly, anomaly_type, anomaly_scores, c_dtw_summary, b4_dtw_summary |
| `continuous_anomaly_report.json` | CUSUM drift detection and 2 s sliding-window results |
| `fatigue_drift_report.json` | 60 s window fatigue monitoring: velocity, asymmetry creep, ROM tightening, percent-fatigue per feature |
| `cross_task_matching.json` | Per-repetition similarity to all Group A reference profiles |
| `dtw_pattern_analysis.json` | Per-repetition DTW distances and shape-anomaly flags |
| `kinematic_summary_group_{A,B,C}.json` | Per-task kinematic summary statistics |

### CSV outputs

| File | Contents |
|------|----------|
| `task_metrics.csv` | Per-task aggregated features (wide format) |
| `repetition_metrics.csv` | Per-repetition features |
| `decision_trace.csv` | Full rule-by-rule traceability: every threshold evaluated |

### Figure outputs (PNG/PDF, 300 dpi)

| File | Contents |
|------|----------|
| `session_summary.pdf` | Academic 7-panel session summary (auto-generated every run) |
| `activation_overlay.pdf` | Multi-page: overlaid repetition traces, mean +/- SD, per task |
| `activation_per_repetition.pdf` | Per-repetition activation detail per task |
| `metrics_summary.png` | Task x metric heatmap |
| `screening_summary.png` | Disorder confidence bar chart |
| `disorder_evidence.png` | Per-disorder evidence contribution waterfall |
| `anomaly_results.pdf` | Deviation heatmap, score histogram, PCA projection, radar chart |
| `anomaly_indication_flow.png` | 3-panel: task x anomaly-type heatmap; type x indication co-occurrence; confidence bars |
| `fatigue_drift_analysis.png` | 4-panel fatigue monitoring figure |
| `asymmetry_over_time.pdf` | L-R asymmetry time series and signed mean per repetition |
| `cross_task_matching.png` | Group A similarity heatmap with substitution cells outlined |
| `dtw_pattern_analysis.pdf` | Temporal shape deviation per task |
| `speech_scores.png` | Unified per-task articulation scores with component heatmap and clinical thresholds |
| `kinematic_profiles_{session}.pdf` | Group B/C spatiotemporal kinematic profiles |
| `kinematic_group_A_profiles.pdf` | Group A blendshape kinematic profiles with +/-2 SD reference overlay |
| `muscle_group_activation_heatmap.png` | Muscle group x task activation grid |
| `anatomical_report.pdf` | Anatomical muscle group analysis report |
| `brain_activation_map.png` | Cortical/subcortical activation map coloured by screening result |
| `recording_cam{N}_annotated.mp4` | Annotated video with landmark overlays |
| `recording_cam{N}_landmarks_only.mp4` | Landmark skeleton on black background |

**Subject-level consolidation outputs** (written to `data/results/{mode}/{subject}/`):

| File | Contents |
|------|----------|
| `{subject}_detection_quality_summary.pdf` | 4-panel recording quality overview across sessions |
| `{subject}_condition_comparison.pdf` | Paired comparison of posture/condition combinations |

#### Session summary PDF: 7-panel layout

Auto-generated at the end of every pipeline run. Can also be run standalone via `tools/session_summary_figure.py`.

| Panel | Contents |
|-------|----------|
| **A: Detection Matrix** | Profiles x disorders grid; severity coded M/S; overall confidence on right |
| **B: Task-Group Evidence** | Forest plot: mean deviation +/-1 SD per task group (A/B/C) per profile |
| **C: Deviation Confidence Ellipses** | Per-repetition scatter (deviation score, log-Mahalanobis); classical and robust MCD 1/2 sigma ellipses |
| **D: Anomaly Fraction and Confidence** | Dual x-axis: percentage anomalous windows (cyan) and detection confidence (orange) |
| **E: Frame Detection Quality** | Box plots of per-frame MediaPipe detection confidence per profile |
| **F: Head Yaw Ellipses** | Scatter of head yaw vs. detection confidence; ellipses anchored on normal profile |
| **G: Head Roll Ellipses** | Scatter of head roll vs. detection confidence; guide lines at 15 degrees (upright) and 75 degrees (supine) |

---

## 10. Methodology

### Feature extraction (`src/feature_extraction.py`, `src/video_processor.py`)

MediaPipe FaceLandmarker provides 52 blendshape coefficients, 478 3D face landmarks, and per-frame detection confidence at up to 30 fps.

**Frame-rate resolution:** The pipeline resolves the active frame rate in this priority order: (1) `grantedFrameRate` from the recording metadata JSON; (2) frame rate from the video container header via OpenCV. The `grantedFrameRate` field, set by the browser at stream acquisition time, is authoritative because browser-recorded MP4/WebM files frequently report incorrect frame rates in their container metadata. This matters because SPARC, LDJ, velocity, and DDK rate are all frame-rate dependent.

Per-frame outputs:
- Blendshape activations (0-1)
- Left-right asymmetry per blendshape pair and eyelid aperture ratio
- Head pose: yaw, pitch, roll from landmark geometry
- Landmark-based mouth geometry: opening, lip action, jaw excursion

Preprocessing (`src/preprocessing.py`) filters frames below a detection confidence threshold and applies brightness normalisation.

#### Head pose estimation (`src/head_pose.py`)

| Angle | Formula | Neutral = 0 degrees |
|-------|---------|-----------------|
| **Yaw** | arctan2(nose_x - eye_centre_x, 0.5 * eye_span) | Face looking straight at camera |
| **Pitch** | arctan2(nose_y_offset - midline/2, face_height) | Nose at midpoint between eyes and mouth |
| **Roll** | arctan2(eye_delta_y, eye_span) | Inter-eye line level for upright recording |

Range (-90, +90) degrees for all angles. Supine/OR recordings give roll approximately +/-40 to 90 degrees.

### Kinematic feature extraction (`src/kinematic_speech.py`)

Follows the blendshape-proxy webcam kinematics approach validated by Palmer et al. [2024] and Simmatis et al. [2023].

**Group A**: per-task features: `kin_a_mean_activation`, `kin_a_peak_amplitude`, `kin_a_velocity`, `kin_a_acceleration`, `kin_a_asymmetry`. Visualised with +/-2 SD reference overlay in `kinematic_group_A_profiles.pdf`.

**Groups B/C**: landmark-based features including mouth opening, lip action, jaw excursion, symmetry, and relative timing.

**DDK clinical metrics** (Group B), following Allison et al. [2022] and Simmatis et al. [2023]:

| Metric | Description |
|--------|-------------|
| `ddk_rate_hz` | Syllable rate (Hz) |
| `ddk_D_mean` | Mean peak-to-peak amplitude (mm-proxy) |
| `ddk_D_max` | Maximum single-cycle amplitude |
| `ddk_Tsd` | Cross-cycle temporal standard deviation (regularity) |
| `ddk_STI` | Spatiotemporal Index (lower = more regular) |
| `ddk_Duration_s` | Total task duration |
| `ddk_Num_Cycles` | Number of full cycles detected |
| `ddk_speed_pct25/50/75/95` | Speed percentile distribution |

**Speech-specific FACS mapping**: 11 action units (AU9, AU11, AU12, AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU26) mapped to MediaPipe blendshapes following Newby et al. [2025].

**Onset timing**: per-repetition `onset_time_s` computed as first frame where activation reaches 25% of peak, following Pantic (2009).

**Articulation smoothness**: two metrics, chosen per task group:

- **Group B (DDK):** Frame-rate-normalised SPARC (spectral arc length) on speech-channel velocity signals. Normalised by (fs / 30) to make the score frame-rate independent. SPARC is well-suited here because irregularity across many rhythmic cycles creates measurable spectral complexity.

- **Group C (word production):** Active-phase LDJ (Log Dimensionless Jerk) with empirically calibrated absolute mapping, blended 60/40 with active-phase SPARC. LDJ and SPARC formulae from Balasubramanian et al. [2012]; speech-specific phase isolation, webcam calibration bounds, and metric blending are project-specific extensions validated against PAC3 and PAC1 healthy-speaker data.

**Known limitation:** MediaPipe FaceLandmarker applies internal Kalman-filter smoothing to blendshapes before data reaches this pipeline. This compresses the practical SPARC range (approximately 0.76-0.86 regardless of true movement quality), limiting the maximum observable deviation across disorder profiles. Disorder detection therefore relies primarily on timing (duration ratio) and amplitude components.

### Anomaly detection (`src/anomaly.py`)

Designed for small reference sets (n = 3-10):

- **Deviation scoring**: t-distribution prediction intervals per feature with proper small-n uncertainty
- **Geometric scoring**: Mahalanobis distance (Ledoit-Wolf shrinkage [Ledoit & Wolf 2004]), nearest-centroid distance, within-session LOF
- **ML scoring**: OC-SVM / IsolationForest, activated only when n_reference >= 10
- **n_ref-adaptive weights**: geometric distances scale linearly from n=3 to n=10
- **Composite score**: z-score (0.45) + fraction deviant (0.30) + IQR fence (0.15) + kinematic (0.10); weights sum to 1.0, output in [0, 1]
- **Multi-feature gate**: a window is anomalous only when 2 or more features simultaneously exceed the z-threshold, preventing single-feature false positives
- **Snippet-based function preservation** (`_snippet_function_preserved`): the session is divided into temporal snippets; function is considered preserved if any snippet contains at least one normal repetition, mirroring the clinical preserved-if-ever-correct criterion [Kanno & Mikuni 2015]
- **Gap bridging**: adjacent anomalous windows separated by 1 s or less are merged

**Anomaly type classification:**

| Type | Meaning |
|------|---------|
| `facial_asymmetry` | L-R asymmetry features deviate |
| `side_amplitude` | Per-side amplitude mismatch |
| `temporal_distortion` | Rate, timing, velocity deviate |
| `articulation` | Phoneme/syllable quality deviation |
| `kinematic_profile` | Shape trajectory deviation |
| `task_substitution` | Cross-task profile mismatch |
| `amplitude_reduction` | Overall amplitude decrease |

### DTW temporal pattern analysis

Detects two temporal anomaly classes independent from amplitude:
- **Initiation delay**: correct pattern but time-shifted (motor initiation slowing)
- **Shape-preserving slowdown**: duration stretched while shape is preserved (linked to dysarthria severity)

Each repetition is time-normalised. DTW (Sakoe & Chiba 1978) distance more than 2 SD above inter-reference variability is flagged. Results in `dtw_pattern_analysis.json` and `dtw_pattern_analysis.pdf`.

### Fatigue and motor drift monitoring (`src/anomaly.py`)

Runs 60 s sliding windows (10 s step) against a 120 s baseline. Three empirical studies ground the design:

| Study | Key finding | Use in pipeline |
|-------|-------------|-----------------|
| Di Stasi et al. [2014] | Saccadic peak velocity decreased after 24 h call shift | Velocity-decay flag: z < -2 AND percent change < -25% |
| Kong et al. [2021] | 21/34 webcam facial indices correlated with PVT performance (r up to 0.89) | Fatigue-risk blendshape set; asymmetry-creep flag |
| Brach & VanSwearingen [1995] | IEMG declines during sustained facial contractions: brow raise 34.51%, smile 22.96% | Grounds the dynamic-range ROM metric |

**Flags per 60 s window:**

| Flag | Condition |
|------|-----------|
| `velocity_decay` | z < -2.0 AND percent change < -25% |
| `asymmetry_creep` | Asymmetry index absolute change from baseline > 0.10 |
| `rom_tightening` | z < -2.0 on intra-window ROM standard deviation |

Outputs: `fatigue_drift_report.json` and `fatigue_drift_analysis.png`.

### Cross-task profile matching (`src/task_profile.py`)

Each Group A test repetition is compared against all nine reference task profiles via z-score similarity. A repetition matching a different task profile more strongly than its own is classified as a **task substitution**: the primary kinematic indicator of buccofacial apraxia. Outputs `substitution_rate` and `task_profile_similarity`. Visualised in `cross_task_matching.png`.

---

## 11. Disorder Screening Logic

Implemented in `src/decision_support.py`, configured in `config/decision_rules.yaml`. Two independent evaluation paths (feature-based and anomaly-based) run in parallel; their indications are merged, with duplicates resolved by keeping the higher-confidence instance. All rule activations are recorded in `decision_trace.csv`.

| Disorder | Evidence |
|----------|----------|
| **Facial paresis** | `mean_asymmetry_ratio > 0.15` sustained across 60% or more of A tasks AND `asymmetry_consistency > 0.70` vs reference baseline; anomaly detector must flag 2 or more Group A tasks (asymmetry or side-amplitude type) |
| **Buccofacial apraxia** | 2 or more qualifying signals: `substitution_rate > 0.20`, `task_profile_similarity < 0.35`, `execution_correctness_score < 0.60`, kinematic_profile anomaly type. Hard gate: `mean_asymmetry_ratio < 0.29` to exclude profiles with notable paresis-induced asymmetry |
| **Dysarthria** | Articulation declined > 0.38 vs reference AND `articulation_impairment_consistency > 0.50`; OR uniform temporal slowdown with `group_b_mean_duration_ratio > 1.20` across 2 or more B tasks |
| **Speech apraxia** | Three independent paths: (1) pa-ta-ka disproportionately declined vs simple syllables; (2) high repetition variability on pa-ta-ka; (3) excess jaw effort on B4 vs baseline. OC-SVM B4 gate requires `b4_vs_simple_dtw_ratio > 1.35` (upright) or `> 1.70` (ORS) |
| **Phonological disorder** | Three independent paths: (1) `c_n_high_relative >= 1` AND `c_dtw_mean > 0.08`; (2) elevated cross-word score variance with quality also declined; (3) extreme amplitude drop on complex C5-C8 words vs reference. Kinematic shift path suppressed when `wpq < 0.74 AND b4_ratio_raw < 1.15` |

---

## 12. Cross-Participant Analysis

`src/cross_participant.py` aggregates outputs across subjects. Run via CLI or Python (see Section 7). Outputs:

- `group_session_overview.csv`: one row per session across all subjects
- `group_aggregated.csv`: aggregated metrics per subject
- `group_boxplots_overview.pdf`: distributional overview
- `group_boxplots_deviation.pdf`: deviation score distributions
- `group_correlation_matrix.pdf`: cross-metric correlation

---

## 13. Subject Consolidation

`src/consolidate.py` runs after all sessions for a subject are processed:

```bash
python -m src.consolidate --subject PAC1 --mode pilot
```

Compares upright vs supine/ORS conditions. Outputs written to `data/results/{mode}/{subject}/`:

- `{subject}_detection_quality_summary.pdf`: 4-panel quality overview across sessions
- `{subject}_condition_comparison.pdf`: paired comparison of posture/condition combinations

---

## 14. Configuration

All thresholds and parameters are in `config/`. No hard-coded values in source files.

| Key | File | Default |
|-----|------|---------|
| `anomaly.composite_threshold` | `decision_rules.yaml` | 0.45 |
| `anomaly_detection.deviation_threshold_std` | `decision_rules.yaml` | 2.0 |
| `anomaly_detection.isolation_forest.contamination` | `decision_rules.yaml` | 0.1 |
| `general.save_dpi` | `plotting.yaml` | 300 |
| Fatigue norms (brow raise, smile) | `features.yaml` | 34.51%, 22.96% |

---

## 15. Dependencies

| Package | Purpose |
|---------|---------|
| mediapipe >= 0.10 | FaceLandmarker blendshapes and 3D landmarks |
| opencv-python | Video decode and camera capture |
| numpy, scipy | Signal processing, statistics, DTW |
| pandas | Feature tables, CSV I/O |
| scikit-learn | OC-SVM, IsolationForest, PCA, Ledoit-Wolf covariance |
| matplotlib | All figures (headless Agg backend) |
| flask | Browser upload UI |
| pyyaml | Configuration loading |
| openpyxl | Reading Excel-format timestamp files |
| pillow | Image processing for figure generation |
| soundfile | Audio extraction for camera sync |

Install via `python3 setup.py` or `pip install -r requirements.txt`.

---

## 16. Key References

Citations are embedded in the relevant source files. A curated list by topic:

### Kinematic feature extraction and DDK

- **Palmer et al. (2024).** Facial Movements Extracted from Video for the Kinematic Classification of Speech. *Sensors* 24(22), 7235. doi:10.3390/s24227235. Core approach in `kinematic_speech.py`.
- **Simmatis et al. (2023).** Analytical Validation of a Webcam-Based Assessment of Speech Kinematics. *Digital Biomarkers* 7(1), 7-17. doi:10.1159/000529685. ICC-A >= 0.70 vs EMA/RealSense; speed percentile features.
- **Allison et al. (2022).** Use of Automated Kinematic DDK Analysis to Identify Potential Indicators of Speech Motor Disorder. *Am J Speech Lang Pathol* 31(6), 2835-2846. doi:10.1044/2022_AJSLP-21-00241. STI and cycle-detection methodology.
- **Balasubramanian et al. (2012).** A robust and sensitive metric for quantifying movement smoothness. *IEEE Trans Biomed Eng* 59(8), 2126-2136. doi:10.1109/TBME.2011.2179545. Introduces SPARC and LDJ as smoothness metrics.
- **Gulde & Hermsdörfer (2018).** Smoothness metrics in complex movement tasks. *Front Neurol* 9:615. doi:10.3389/fneur.2018.00615. Grounds the Group B/C metric split.
- **Newby et al. (2025).** The Role of Facial Action Units in Investigating Facial Movements During Speech. *Electronics* 14(10), 2066. doi:10.3390/electronics14102066. FACS AU-to-blendshape mapping for 11 speech-relevant action units.
- **Pantic (2009).** Machine analysis of facial behaviour. *Phil Trans R Soc B* 364, 3505-3513. doi:10.1098/rstb.2009.0135. 25%-of-peak onset marker.
- **Lucero & Munhall (2008).** Analysis of Facial Motion Patterns During Speech. *J Acoust Soc Am* 124(4), 2283-2290. doi:10.1121/1.2973196. Coupled-oscillator DDK regularity framing.

### Facial asymmetry and paralysis detection

- **Heinrich et al. (2025).** Deep learning-based automatic facial symmetry scoring in peripheral facial palsy. *Sci Rep* 15. doi:10.1038/s41598-025-17172-1. MediaPipe-based landmark asymmetry; mouth-corner features most sensitive.
- **Oliveira et al. (2024).** Facial expressions to identify post-stroke: A pilot study. *Comput Methods Programs Biomed* 250, 108195. doi:10.1016/j.cmpb.2024.108195.
- **Ozmen et al. (2025).** Development of a Novel Multi-feature ML Model for Unilateral Facial Paralysis. *Plast Reconstr Surg*. doi:10.1097/01.GOX.0001112148.28567.85.
- **Baig et al. (2023).** Facial Paralysis Recognition Using Face Mesh-Based Learning. University of Johannesburg. hdl:10210/504453.
- **Ruiter et al. (2023).** Assessing facial weakness in myasthenia gravis with facial recognition software. *Ann Clin Transl Neurol* 10(8), 1314-1325. doi:10.1002/acn3.51823.
- **Alagha et al. (2023).** Mathematical Validation of the Modified Sunnybrook Facial Grading System. *J Plast Reconstr Surg* 2(3), 77-88. doi:10.53045/JPRS.2022-0017.
- **Ross et al. (1996).** Development of a sensitive clinical facial grading system. *Otolaryngol Head Neck Surg* 114(3), 380-386. doi:10.1016/S0194-5998(96)70206-1.

### Intraoperative monitoring and awake craniotomy

- **Collée et al. (2022).** Speech and Language Errors during Awake Brain Surgery and Postoperative Language Outcome. *Cancers* 14(21), 5466. doi:10.3390/cancers14215466.
- **Collée et al. (2023).** Localization patterns of speech and language errors during awake brain surgery: a systematic review. *Neurosurg Rev* 46. doi:10.1007/s10143-022-01943-9.
- **Kanno & Mikuni (2015).** Evaluation of Language Function under Awake Craniotomy. *Neurol Med Chir* 55(5), 367-373. doi:10.2176/nmc.ra.2014-0395. Clinical rationale for preserved-if-ever-correct criterion.
- **De Witte et al. (2015).** The Dutch Linguistic Intraoperative Protocol. *Brain Lang* 140, 35-48. doi:10.1016/j.bandl.2014.10.011.
- **Frajtag et al. (2025).** Evaluation of Facial Landmark Localization Performance in a Surgical Setting. *Mechanisms and Machine Science* 190, 278-287. doi:10.1007/978-3-032-02106-9_31.

### Computer vision and multi-camera processing

- **Kitaguchi et al. (2022).** Artificial intelligence-based computer vision in surgery: Recent advances and future perspectives. *Ann Gastroenterol Surg* 6(1), 29-36. doi:10.1002/ags3.12513.
- **Lugaresi et al. (2019).** MediaPipe: A Framework for Perceiving and Processing Reality. CVPR Workshop on Computer Vision for AR/VR.
- **Kartynnik et al. (2019).** Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs. arxiv:1907.06724.
- **Scott et al. (2022).** Healthcare applications of single camera markerless motion capture: a scoping review. *PeerJ* 10. doi:10.7717/peerj.13517.

### Anomaly detection and fatigue monitoring

- **Ledoit & Wolf (2004).** A well-conditioned estimator for large-dimensional covariance matrices. *J Multivariate Anal* 88(2), 365-411. doi:10.1016/S0047-259X(03)00096-4.
- **Sakoe & Chiba (1978).** Dynamic Programming Algorithm Optimization for Spoken Word Recognition. *IEEE Trans Acoust Speech Signal Process* 26(1), 43-49. doi:10.1109/TASSP.1978.1163055.
- **Di Stasi et al. (2014).** Saccadic Eye Movement Metrics Reflect Surgical Residents' Fatigue. *Ann Surg* 259(4), 824-829. doi:10.1097/SLA.0000000000000260.
- **Kong et al. (2021).** Facial Features and Head Movements Obtained with a Webcam Correlate with Performance Deterioration. *Atten Percept Psychophys* 83(1), 525-540. doi:10.3758/s13414-020-02199-5.
- **Brach & VanSwearingen (1995).** Measuring Fatigue Related to Facial Muscle Function. *Arch Phys Med Rehabil* 76(10), 905-908. doi:10.1016/S0003-9993(95)80064-6.
