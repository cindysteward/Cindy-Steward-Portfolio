# Study Prompter: Clinical Speech & Facial Monitoring (for FACET-CV)

A self-contained HTML tool that guides participants through structured facial expression and speech tasks for clinical research. The researcher configures a session (language, profile, recording options), and the tool walks the participant through the selected task profile with animated SVG demonstrations and text-to-speech instructions.

## Quick Start

1. Open `study-prompter.html` in **Google Chrome** (recommended for full API support).
2. Enter a Participant ID, select a profile and language (Dutch or English).
3. Optionally pick a specific TTS voice and preview it.
5. Optionally set Sequence Repetitions (A & C) in the setup screen (default 3, B always 5).
6. Optionally select a single task block (A, B, or C) instead of running all three.
5. Optionally enable camera and/or screen recording. For camera recording, enable the toggle, click **Scan for cameras**, then click **Grant Access** next to each camera slot you want to record. Chrome will show a camera picker popup once per slot. After granting, the checkbox is enabled — check it to include that camera.
6. Click **Begin Session**.

## Changes (2026-03-25)

- Section A description no longer includes a "no need to rush" sentence.
- Two distinct smile tasks: **"Smile" / "Glimlachen"** (gentle smile, no teeth; new `smile_gentle` expression) and **"Smile wide, show your teeth" / "Lach breed, laat uw tanden zien"** (wide smile showing teeth; uses the existing `happy` expression).
- Section B: each articulation item is now a single task that lasts 5 seconds where the participant performs the sequence 5× as fast as possible. Dysarthria variants are presented as a single 5-second trial saying the sequence slowly and heavily 5×. The inner repetition loop has been removed.
- Section C: words are presented by audio only (the word is not shown on screen). The on-card instruction now reads: "Repeat the word you hear" / "Herhaal het woord dat u hoort".
- The GET READY cue now displays immediately and animates an inner ready-bar while the TTS instruction plays. The GO signal appears the instant the TTS finishes and the task starts; the GO badge hides after 600 ms.
- Camera recordings are fault-tolerant: if a camera recorder stopped mid-session, any recorded chunks from that camera are still saved and offered for download at session end.
- Default task duration is now 3 seconds for Sections A and C; Section B tasks are always 5 seconds.

No build tools, servers, or installations required — just open the file.

## Features

- **Language selection**: choose Dutch (Nederlands) or English; all UI text, instructions, and TTS adapt to the selected language
- **Voice selection**: pick from any system TTS voice matching the selected language, with a preview button to hear it before starting; defaults to automatic Auto (Google neural preferred) selection
- **Task block selection**: optionally run only Block A (facial), Block B (speech), or Block C (words) instead of all three; baseline always runs regardless of selection
- **Optional sequence repetition override**: set custom reps for Sections A and C on the setup screen (default 3 each; Section B remains fixed at 5; ignored in COMBINED mode)
- **10 clinical profiles** covering baseline, facial paresis, buccofacial apraxia, dysarthria, speech apraxia, phonological disorder, mixed conditions, and a COMBINED profile that runs fully deduplicated tasks covering every disorder (17 Section A expressions × 3 reps, 11 Section B items × 5 reps, 28 Section C words × 3 reps) with per-task `profile_label` annotations using standardised profile keys (e.g. `NORMAL,P1_PARESIS,MIXED_C`); the `profile_label` badge is never shown to the participant
- **Practice round**: a short optional practice before the baseline lets participants try one facial, one speech, and one word task to get familiar; practice tasks appear in the CSV with section `Practice` and do not affect session scoring
- **Neutral baseline measurement**: a 10-second neutral baseline is captured at the start of every session before the selected task blocks begin; the baseline section break hides the title and shows larger white description text, then auto-continues after TTS with a Netflix-style fill button.
- **Detailed SVG face**: narrower egg-shaped head with a 52px-wide neck; no hair (clean shaved head); almond-shaped eyes clipped via SVG `clipPath` (top arc peaks at y=190, bottom dips to y=228), with eye centres at y=209; larger sclera (rx=28 ry=18), iris (r=12), and pupil (r=5.5); one filled skin-tone upper eyelid per eye peaking at y=190; eyebrows spanning the full brow ridge with correct `transform-box: fill-box` so they rotate about their own centre during frown expressions; permanent low-opacity warm blush circles on the cheeks with separate animated blush for happy; smile with raised corners at (140,290)/(260,290), thin upper lip (outer edge peaks at y=287, inner edge at y=293, stroke-width 0.8, minimal gap to teeth), and lower lip; nasolabial folds and dimples anchoring to the same corner points; C-shaped ear shapes, nose bridge and nostrils, cheek puff overlays, furrow lines
- **Smooth expression animations**: crossfade between mouth shapes with coordinated eyebrow, cheek, eye, and lid animations via anime.js
- **Asymmetric paresis expressions**: two dedicated SVG mouth groups (`m-asym-teeth`, `m-asym-puff`) simulate left-sided facial weakness. Asymmetric teeth now renders a curved right-sided grin with visible teeth and no dark gap, and asymmetric puff inflates only the right cheek while the left stays relaxed; PARESIS profile tasks 3 and 8 use these expressions
- **Tongue animations**: tongue emerges from mouth cavity with proper overlap; left-right timeline loop has smooth eye-tracking; tongue-up and tongue-out connect naturally to the open mouth shape
- **GO! timing**: the task timer and CSV `time_from_start_s` timestamp fire the instant GO! appears on screen, not 600 ms later when it fades. The GO! badge stays visible for 600 ms as a visual holdover, but the task clock starts immediately.
- **GET READY cue**: replaced the old numeric countdown with an animated 1.4-second progress bar under the GET READY text; GO! is displayed for 600 ms before starting the task.
- **Recording timestamp alignment**: `time_from_start_s` in the CSV is measured from the moment the first recorder started (before TTS announcements), not from when the session timer UI started. The `recording_start_offset_s` field in the metadata JSON records how many seconds of pre-session video precede the first logged task event.
- **Start/stop visual cues**: a multi-step countdown badge (GET READY → 3 → 2 → 1 → GO!) appears before every task at 42px bold text; GET READY is red, the countdown digits are orange, and GO! is green; the badge is large enough to read at a glance from across the room; a colour-coded progress bar accompanies the cue.
- **Hold instructions**: section break screens tell participants to begin on the GO! cue and hold until the timer completes
- **Section break TTS**: bridging announcements like "Section A: facial expression tasks" or "Practice round" are no longer spoken; only the participant-facing section description is spoken, followed by Netflix-style auto-continue after a fill bar.
- **Clean transitions**: previous task instructions and expression state are cleared when transitioning between sections, ensuring a clean slate for each block
- **Speech & word tasks show text only**: the SVG face is hidden during Sections B and C. Each word card now displays a small uppercase action label above the target word (e.g. Say / Repeat / Say slowly and heavily) and the word itself is shown in large text.
- **Smart TTS pacing**: slow/heavy instructions split into two utterances — the instruction verb (e.g. "Slowly repeat" / "Say slowly and heavily" / "Herhaal langzaam" / "Zeg langzaam en zwaar") is spoken at normal rate, then only the actual syllable or word content is spoken at slow rate (pitch 0.68, rate 0.34, volume 1.0); normal syllable-sequence tasks (pa-pa-pa, ta-ta-ta, ka-ka-ka, pa-ta-ka) use a faster TTS rate (`CONFIG.ttsRateFast`, default 0.92); all other TTS uses pitch 1.1 (Dutch) or 1.05 (English) and rate 0.80 (Dutch) or 0.78 (English) from `CONFIG.ttsVolume`; clinical syllable sequences are phonetically normalised (pah/tah/kah) with `...` separators for natural inter-syllable pauses (hyphened form uses single ellipsis: `pah... pah... pah`; dotted form uses double: `pah...... pah...... pah`); instruction verbs (Repeat/Say/Herhaal/Zeg) get a brief pause after the colon; typographic characters (ellipsis, em dash) normalised for natural delivery; Dutch voice selection prefers exact `nl-NL` locale before falling back to any `nl` voice; soft-preference list includes xander, anna, and fem-prefixed voices
- **Frown expression**: brows drop (translateY 14) and rotate inward (±28°) using CSS `transform-box: fill-box` so the pivot is each brow's own centre; the frown mouth is wider with corners at y=312 and centre at y=296 for a more pronounced downward curve; asymmetric right-frown applies only to the right brow; expression reset is delayed 700 ms to allow animations to complete before advancing to the next task
- **Section A task instructions**: shortened to be as concise as possible (e.g. “Tongue: left corner, then right corner”, “Raise eyebrows: surprised”, “Frown: right brow more”); PARESIS and MIXED_A/MIXED_B/MIXED_C use positive directional phrasing (“right side more” instead of “left less”); COMBINED_A mirrors all same strings
- **Speech and word task timing**: `showSpeechWordTask` now plays TTS first and starts the countdown only after TTS finishes, consistent with how baseline and facial tasks already behave
- **Camera recording** uses a slot-based permission model: the researcher clicks *Grant Access* on each camera slot, which triggers Chrome's native camera picker popup for that slot so they can select a specific physical camera. Up to 4 slots can be added. Granted slots persist across re-scans. Virtual cameras (e.g. OBS Virtual Camera, ManyCam) are excluded automatically when real labels are known; OBSBOT and other hardware cameras with "OBS" in their name are not filtered. Camera streams are recorded at up to 4K 60 fps with 25 Mbps bitrate; the actual granted resolution and frame rate are stored in the metadata JSON.
- **Screen recording** captures the app window at WebM/VP9 8 Mbps (avoids the frozen-first-frame bug that affects MP4/H.264 screen capture in Chrome) with a mixed microphone audio track so that TTS output and participant speech are both audible in the recording. The screen stream and microphone stream are acquired when the toggle is turned on (before the session starts) and released automatically when the session ends or the toggle is turned off. If the selected microphone is unavailable, screen recording continues without audio.
- **Researcher controls**: pause, skip, or end the session at any time via on-screen buttons or keyboard shortcuts (Space = pause/resume, → = skip task, Esc = end session); Skip immediately cancels any in-progress countdown
- **Auto-download** of recordings, a recording metadata JSON (per-camera resolution/framerate/bitrate/offset), and a timestamped CSV of all task start/end events with participant ID, profile, session date, section, task number, sequence, task type, label, `profile_label`, expression, event (start/end), `time_from_start_s` (elapsed from recording start), wall time, and `sequence_rep` (which repetition of the full sequence this task belonged to). The `profile_label` column uses standardised comma-separated profile keys (e.g. `NORMAL,P1_PARESIS,MIXED_C`) to identify the disorder source for each task in the COMBINED profile.
- **Setup screen**: two-column grid layout (860 px max-width) with scrollable body; Participant ID, Task Blocks, and Task Duration are in the left column; Profile and Language are in the right column; Voice, Recording Options, and Begin Session span full width; collapses to a single column on screens ≤700 px

## Session Structure

Each profile runs a baseline plus up to three sections (configurable via the Task Blocks selector):

| Phase | Content | Details |
|-------|---------|---------|
| **Practice** | 1 facial + 1 speech + 1 word task | Always runs first; labelled as practice; does not count toward session data |
| **Baseline** | Neutral expression | 10 seconds, face shown (always runs) |
| **A**: Facial | 9 expression tasks with animated face | Full sequence × 3, 3 sec each (NORMAL); disordered profiles run each task once. COMBINED: 17 deduplicated expressions × 3 sequences |
| **B**: Speech | 4 syllable repetition tasks (text only) | All profiles: all 5 repetitions of each item are grouped together (pa-pa-pa × 5, then ta-ta-ta × 5, etc.); if the item has sequencing variants (apraxia), all 3 wrong-order variants are shown as separate consecutive tasks. COMBINED: 11 deduplicated items × 5 repetitions each, item-first order |
| **C**: Words | 8 word repetition tasks (text only) | All profiles: full word list repeated × 3 sequences with rest breaks between. COMBINED: 28 deduplicated words × 3 sequences |

By default all blocks run. Selecting a single block runs only that block after the baseline.

## Profiles

| Key | Name |
|-----|------|
| NORMAL | Baseline |
| PROFILE 1 | Facial Paresis (Left-sided) |
| PROFILE 2 | Buccofacial Apraxia |
| PROFILE 3 | Dysarthria |
| PROFILE 4 | Speech Apraxia |
| PROFILE 5 | Phonological Disorder |
| MIXED A | Speech Apraxia + Mild Facial Asymmetry |
| MIXED B | Dysarthria + Minor Buccofacial Noise |
| MIXED C | Facial Paresis + Phonological Disorder |
| COMBINED | All Disorders (Minimum Time) |

## Recordings

When recording is enabled, files auto-download on session end:

- **Camera (single)**: `[PID]_cam_[YYYY-MM-DD_HHMMSS].mp4` (or `.webm` if MP4 not supported)
- **Camera (multiple slots)**: `[PID]_cam1_[timestamp].mp4`, `[PID]_cam2_[timestamp].mp4`, … → each file includes selected microphone audio
- **Screen**: `[PID]_screen_[timestamp].webm` (WebM/VP9; mixed microphone + tab audio)
- **Recording metadata JSON**: `[PID]_recording_meta_[timestamp].json` — one entry per camera with granted resolution, frame rate, MIME type, bitrate, and `recording_start_offset_s` (seconds of pre-session video before the first logged task)

  Each entry in the metadata JSON corresponds to one camera and contains: `deviceLabel` (the physical camera name as reported by Chrome, e.g. "FaceTime HD Camera" or "Elgato Facecam"), `camera_index` (1-based integer matching the `cam1`, `cam2` suffix in the video filename), and `start_offset_from_first_cam_s` (seconds between this camera starting and the first camera starting — used by the analysis pipeline to align video timelines). For the first camera this offset is always 0.0.
- **Timestamps CSV**: `[PID]_timestamps_[timestamp].csv` — columns: `participant_id`, `profile`, `session_date`, `section`, `task_number`, `sequence`, `task_type`, `label`, `profile_label`, `expression`, `event` (start/end), `time_from_start_s` (elapsed since recording started), `wall_time`, `sequence_rep`
- **Assembly CSV** (COMBINED profile only): `[PID]_assembly_[timestamp].csv` — expands each COMBINED task into one row per disorder profile listed in `profile_label`; columns: `disorder_profile`, `section`, `task_number`, `sequence_rep`, `label`, `expression`, `start_s`, `end_s`, `wall_time`, `participant_id`, `session_date`; rows sorted by profile → section → task → rep, with a blank separator row between each profile group

> `time_from_start_s` counts from the moment recording began (before the session TTS announcement). Use `recording_start_offset_s` from the metadata JSON to locate the exact frame in the video corresponding to t=0 in the CSV.

At session end, all files are listed in a panel on the completion screen.
Each file shows its download status and two action buttons: **Re-download**
(sends the file to the browser's default Downloads folder again) and
**Save to…** (opens a native file picker so the file can be saved to any
location, including an external drive or network share). Use **Save to…**
if the default download location ran out of storage or a file was not saved
correctly. Files remain available in the panel until the next session begins.
A storage warning appears automatically if available space is low.
The **Save to…** button requires Chrome 86 or later and is shown only when
the browser supports the File System Access API.

## Browser Requirements

- **Google Chrome 90+** (recommended)
- Camera/microphone permissions required if camera recording is enabled
- Screen sharing permission required if screen recording is enabled
- Internet connection required on first load (anime.js CDN)

## External Dependency

- [anime.js 3.2.2](https://animejs.com/):  loaded from CDN for SVG animation

## File Structure

```
tasklist_animation/
├── study-prompter.html   # Complete self-contained application
├── README.md             # This file
```

## File Intake

`intake.py` groups downloaded session files by participant ID and timestamp, then
moves each group into a structured folder ready for pipeline processing.

```
python intake.py                              # scan current folder
python intake.py --src ~/Downloads            # scan a specific folder
python intake.py --src ~/Downloads --dst ~/study_data/raw_sessions
python intake.py --src ~/Downloads --dry-run  # preview without moving

example: python intake.py --src ~/Downloads --dst ~/path/to/tasklist_animation/sessions

python intake.py --src ~/Downloads --audio-shift-ms 80 # add shift lag if there is a fixed offset at the start (audio always early or late)
```

Output structure:

```
<dst>/<pid>/<pid>_<timestamp>/
    <pid>_cam1_<timestamp>.webm
    <pid>_screen_<timestamp>.webm   (if present)
    <pid>_timestamps_<timestamp>.csv
    <pid>_assembly_<timestamp>.csv  (COMBINED sessions only)
    <pid>_recording_meta_<timestamp>.json
```

After moving the files, the script prints the `run_pipeline.py` command for each
session with all paths pre-filled (the `--session` label is left as a placeholder
for the researcher to fill in).  A summary table at the end lists how many sessions
were found, how many files were moved, and any files that did not match the expected
naming pattern.

## Session Label Naming Conventions

The analysis pipeline connects the study prompter output to the pipeline through one
field you choose at analysis time: the **session label**, passed as `--session` on
the command line or typed into the Session Label field in the web UI. This label is
not read from any file, it is yours to assign. It determines two things: where
output folders are created, and whether the session is treated as a reference
(baseline) or a test session.

**Why the label matters for reference sessions**

The pipeline auto-detects whether a session is a reference session by checking if
the label contains the word `baseline` or `normal` (case-insensitive). Reference
sessions are used to train the anomaly detector and normalise features for all later
test sessions for the same participant. If none of these words appear, the session is
treated as a test session and the pipeline will search existing sessions to find a
reference automatically. If your label does not follow this convention for a NORMAL
profile run, the session will not be registered as a reference and comparison will
fail or fall back to self-contained mode with a logged warning.

**Recommended conventions**

Use short, lowercase, underscore-separated labels that describe the condition
clearly. The word `baseline` or `normal` must appear for reference sessions.

| Study type | Profile used | Recommended label |
|---|---|---|
| Pilot — reference | NORMAL | `baseline_upright` |
| Pilot — reference | NORMAL | `baseline_supine` |
| Pilot — reference | NORMAL | `baseline_sim_or` |
| Pilot — test | COMBINED | `combined_upright` |
| Pilot — test | COMBINED | `combined_supine` |
| Patient — pre-op reference | NORMAL | `pre_op_baseline` |
| Patient — intra-op reference | NORMAL | `intra_op_baseline` |
| Patient — intra-op test | COMBINED | `intra_op_t1` |
| Patient — post-op test | COMBINED | `post_op_t1` |

You can use any label you like as long as the convention is respected.
`normal_sitting`, `baseline_condition_a`, and `pre_op_normal` all work.
`session_1` or `upright` do not register as reference sessions.

**Participant ID**

The Participant ID you enter in the study prompter (e.g. `P001`, `PAT003`) becomes
the prefix of every output file. Use the same ID as the `--subject` argument when
running the pipeline. Consistency matters: the pipeline groups all sessions by
subject ID when consolidating results and auto-discovering reference sessions.

The intake helper (`intake.py`) reads the participant ID from the filename prefix
automatically and groups files accordingly.

**Session ID (auto-generated)**

You never set the session ID directly. The pipeline generates it by combining your
`--subject`, `--session` label, and a timestamp:

```
P001_baseline_upright_20260101_120000
```

This becomes the folder name for all output files for that run. If you run the same
subject and label twice, each run gets its own timestamped folder, so nothing is
overwritten.

**Listing existing sessions**

To see all sessions recorded for a subject and find their session IDs for use as
`--reference`:

```bash
python run_pipeline.py --list-sessions --subject P001 --mode pilot
```

## Run with setup.sh

To launch quickly:

1. Open a terminal and go to the folder:
   cd tasklist_animation
2. Run:
   ./setup.sh

This opens the app in Chrome and creates output folders automatically.
