"""
Main application for the facial expression mimicry game.
Handles camera input, expression cycling, hold-timer logic, and display updates.
Prefers an external camera if one is detected, otherwise falls back to the built-in webcam.

Run: python main.py
(Run setup.py first to install dependencies)
"""

import cv2
import sys
import time
import random
import numpy as np

from face_mesh import FaceMeshDetector
from expression_logic import ExpressionDetector, EXPRESSIONS
from ui import create_display_window, assemble_frame
from utils import (
    WINDOW_TITLE, CAMERA_ERROR_MESSAGE,
    HOLD_DURATION, WELLDONE_DISPLAY_TIME,
)


PLAYING = "playing"
CALIBRATING = "calibrating"
WELL_DONE = "well_done"


class ExpressionCycler:
    """Shuffles and cycles through the full list of expressions continuously."""

    def __init__(self, expressions):
        self.all_expressions = list(expressions)
        self.queue = []
        self.current_index = 0
        self._shuffle_new_round()

    def _shuffle_new_round(self):
        self.queue = list(self.all_expressions)
        random.shuffle(self.queue)
        self.current_index = 0

    @property
    def current(self):
        return self.queue[self.current_index]

    @property
    def display_index(self):
        return self.current_index + 1

    @property
    def total(self):
        return len(self.queue)

    def advance(self):
        self.current_index += 1
        if self.current_index >= len(self.queue):
            self._shuffle_new_round()


def find_camera():
    """Try to open an external camera first, then fall back to the built-in webcam."""
    for idx in [1, 2, 0]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Camera found at index {idx}")
                return cap
            cap.release()
    return None


class ExpressionMimicApp:
    """Main application class managing the game loop and state transitions."""

    def __init__(self):
        self.cap = None
        self.face_detector = None
        self.expression_detector = None
        self.cycler = ExpressionCycler(EXPRESSIONS)
        self.state = CALIBRATING
        self.hold_start = None
        self.well_done_start = None
        self.running = False

    def initialize(self) -> bool:
        """Set up camera, detector, and display window."""
        self.cap = find_camera()
        if self.cap is None:
            self._show_error(CAMERA_ERROR_MESSAGE)
            return False

        self.face_detector = FaceMeshDetector()
        self.expression_detector = ExpressionDetector()
        create_display_window(WINDOW_TITLE)
        self.running = True
        return True

    def _show_error(self, message: str):
        """Display an error message briefly and exit."""
        print(f"ERROR: {message}", file=sys.stderr)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
        cv2.putText(frame, message, (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        create_display_window(WINDOW_TITLE)
        cv2.imshow(WINDOW_TITLE, frame)
        cv2.waitKey(3000)

    def run(self):
        """Run the main game loop until the user quits."""
        if not self.initialize():
            return

        print("Starting game loop...")
        frame_fail_count = 0

        while self.running:
            ret, raw_frame = self.cap.read()
            if not ret:
                frame_fail_count += 1
                if frame_fail_count > 30:
                    print("Camera stopped providing frames.", file=sys.stderr)
                    break
                time.sleep(0.03)
                continue
            frame_fail_count = 0

            raw_frame = cv2.flip(raw_frame, 1)
            face_detected, landmarks = self.face_detector.detect_face(raw_frame)
            now = time.time()

            progress = 0.0
            show_well_done = False
            calibrating = self.expression_detector.calibrating
            cal_progress = self.expression_detector.calibration_progress

            if self.state == CALIBRATING:
                if face_detected:
                    calibrated, _ = self.expression_detector.update(landmarks, "")
                    cal_progress = self.expression_detector.calibration_progress
                    if calibrated:
                        self.state = PLAYING

            elif self.state == PLAYING:
                if face_detected:
                    _, detected = self.expression_detector.update(
                        landmarks, self.cycler.current
                    )
                    if detected:
                        if self.hold_start is None:
                            self.hold_start = now
                        elapsed = now - self.hold_start
                        progress = min(elapsed / HOLD_DURATION * 100, 100)
                        if progress >= 100:
                            self.state = WELL_DONE
                            self.well_done_start = now
                            progress = 100
                    else:
                        self.hold_start = None
                        progress = 0
                else:
                    self.hold_start = None
                    progress = 0

            elif self.state == WELL_DONE:
                show_well_done = True
                progress = 100
                if now - self.well_done_start >= WELLDONE_DISPLAY_TIME:
                    self.cycler.advance()
                    self.expression_detector.reset_smoothing()
                    self.hold_start = None
                    self.state = PLAYING

            display = assemble_frame(
                camera_frame=raw_frame,
                face_detected=face_detected,
                landmarks=landmarks if face_detected else None,
                expression_name=self.cycler.current,
                progress_percent=progress,
                show_well_done=show_well_done,
                calibrating=(self.state == CALIBRATING),
                calibration_progress=cal_progress,
                expression_index=self.cycler.display_index,
                total_expressions=self.cycler.total,
            )

            cv2.imshow(WINDOW_TITLE, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.running = False
            elif key in (ord("r"), ord("R")):
                self.expression_detector.recalibrate()
                self.state = CALIBRATING
                self.hold_start = None
            elif key in (ord("s"), ord("S")):
                if self.state == PLAYING:
                    self.cycler.advance()
                    self.expression_detector.reset_smoothing()
                    self.hold_start = None

        self.cleanup()

    def cleanup(self):
        """Release all resources."""
        if self.face_detector:
            self.face_detector.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    ExpressionMimicApp().run()


if __name__ == "__main__":
    main()
