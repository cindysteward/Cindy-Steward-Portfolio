# Expression Mimic

A real-time facial expression mimicry game. The app challenges you to reproduce a series of facial expressions using your webcam and live feedback. An Apple emoji on the left shows the target expression, while the right side displays your camera feed with a landmark overlay. Hold the expression accurately for 2 seconds to advance. The game cycles through all expressions continuously until you quit.

## Features

- **11 Facial Expressions**: Smile, Big Grin, Left Wink, Right Wink, Raised Eyebrows, Kiss Face, Squint, Surprised, Wide Eyes, Jaw Drop, and Tongue Out
- **Live Webcam Feed**: with facial landmark overlay (contour, eyes, eyebrows, mouth)
- **Emoji Prompts**: each target expression is shown as an Apple emoji on the left panel for clear, intuitive guidance
- **2-Second Hold Bar**: when the expression is detected, the progress bar fills over 2 seconds; release and it resets
- **"Well Done!" Celebration**: shown briefly when you complete an expression, then the next one appears
- **Auto-Calibration**: app calibrates to your neutral face within about one second for accurate, per-user detection
- **Recalibrate Anytime**: press R to recalibrate for a new user during the same session without restarting
- **Skip Expression**: press S to skip any expression you find difficult or isn't registering well
- **Continuous Play**: shuffles and cycles through all 11 expressions. Reshuffles after each full round
- **External Camera Preference**: if an external USB camera is detected, it is used automatically; otherwise falls back to the built-in webcam

## Requirements

- **Python 3.10**
- **Webcam** (built-in or external)
- macOS or Windows

## Installation

1. Navigate to the project directory:

   ```bash
   cd "brain awareness week/facial_expression_detector"
   ```

2. Run the setup script:

   ```bash
   python3.10 setup.py
   ```

   This creates a virtual environment and installs all dependencies.

## Usage

1. Activate the environment:

   ```bash
   source venv3.10/bin/activate
   ```

2. Run the application:

   ```bash
   python main.py
   ```

### Controls

- **Q** or **ESC** — Quit the application
- **R** — Recalibrate (for switching users mid-session)
- **S** — Skip the current expression

## How It Works

1. On launch, the app asks you to look at the camera with a relaxed face for about one second (**calibration**). It records your neutral baseline measurements (mouth width, eye openness, brow height, etc.). Press **R** at any time to recalibrate for a different user.
2. A random expression is shown as an Apple emoji on the left, with its name below. Your camera feed appears on the right with facial landmarks drawn on top.
3. When the app detects you are performing the target expression, a progress bar at the bottom begins filling. You must hold the expression steadily for 2 seconds to fill the bar to 100%.
4. Once the bar is full, a "Well Done!" overlay appears briefly, and the game advances to the next expression.
5. After all 11 expressions have been completed, the list is reshuffled and the cycle repeats.

## Project Structure

```text
facial_expression_detector/
├── main.py               # Application entry point and game loop
├── face_mesh.py          # MediaPipe face detection wrapper
├── expression_logic.py   # Expression detection heuristics with calibration
├── ui.py                 # UI rendering, emoji prompts, and overlays
├── utils.py              # Constants, helpers, and shared utilities
├── setup.py              # Environment setup and dependency installation
├── requirements.txt      # Python dependencies
├── __init__.py           # Package metadata
└── README.md             # This file
```

## Troubleshooting

### Camera Not Found

- Ensure the webcam is connected and not in use by another application.
- If you have multiple cameras, the app tries index 1 and 2 (external) before falling back to 0 (built-in). If your camera is on a different index, adjust `find_camera()` in `main.py`.

### Expression Not Detected

- Make sure your face is well-lit and clearly visible to the camera.
- During calibration, keep a neutral, relaxed expression. The detection is relative to this baseline.
- Press S to skip any expression that proves too difficult.
