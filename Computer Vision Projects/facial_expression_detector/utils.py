"""
Shared constants, helper classes, and utility functions for the facial expression mimicry application.
"""

import os
from collections import deque
from typing import Tuple, Deque


WINDOW_TITLE = "Expression Mimic"
TITLE_TEXT = "Mimic This Expression!"
NO_FACE_MESSAGE = "No face detected \u2014 look at the camera"
CAMERA_ERROR_MESSAGE = "Camera unavailable. Please check your camera settings."
CALIBRATING_MESSAGE = "Look at the camera with a relaxed face\u2026"

DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 700
LEFT_PANEL_WIDTH = 260
TITLE_BAR_HEIGHT = 55
BOTTOM_BAR_HEIGHT = 70

TITLE_FONT_SIZE = 24
EXPRESSION_NAME_FONT_SIZE = 20
STATUS_FONT_SIZE = 15
WELLDONE_FONT_SIZE = 44
NO_FACE_FONT_SIZE = 17
HINT_FONT_SIZE = 12
COUNTER_FONT_SIZE = 13

HOLD_DURATION = 2.0
WELLDONE_DISPLAY_TIME = 1.5
CALIBRATION_FRAMES = 20
SMOOTHING_WINDOW = 5

EXPRESSION_EMOJIS = {
    "Smile": "😊",
    "Big Grin": "😁",
    "Left Wink": "\U0001f61c",
    "Right Wink": "😉",
    "Raised Eyebrows": "🤨",
    "Kiss Face": "😘",
    "Squint": "😑",
    "Surprised": "😮",
    "Wide Eyes": "😳",
    "Jaw Drop": "😱",
    "Tongue Out": "😛",
}

COLOR_BG = (245, 243, 240)
COLOR_PANEL_BG = (255, 255, 255)
COLOR_TITLE_BAR = (44, 62, 80)
COLOR_ACCENT = (52, 152, 219)
COLOR_SUCCESS = (39, 174, 96)
COLOR_TEXT_PRIMARY = (50, 50, 50)
COLOR_TEXT_SECONDARY = (140, 140, 140)
COLOR_TEXT_MUTED = (190, 190, 190)
COLOR_BAR_BG = (230, 230, 230)
COLOR_FACE_OUTLINE = (140, 140, 140)
COLOR_FACE_FEATURE = (60, 60, 60)
COLOR_FACE_MOUTH = (180, 60, 60)
COLOR_FACE_CHEEK = (255, 190, 170)

BGR_TITLE_BAR = (80, 62, 44)
BGR_ACCENT = (219, 152, 52)
BGR_SUCCESS = (96, 174, 39)
BGR_BG = (240, 243, 245)
BGR_PANEL_BG = (255, 255, 255)
BGR_BAR_BG = (230, 230, 230)
BGR_LANDMARK_EYE = (255, 255, 0)
BGR_LANDMARK_BROW = (200, 60, 120)
BGR_LANDMARK_MOUTH = (60, 60, 200)
BGR_LANDMARK_CONTOUR = (200, 200, 200)
BGR_DIVIDER = (220, 220, 220)


class RollingAverage:
    """Maintains a rolling average over a fixed-size window of values."""

    def __init__(self, window_size: int = SMOOTHING_WINDOW):
        self.values: Deque[float] = deque(maxlen=window_size)

    def add(self, value: float) -> float:
        self.values.append(value)
        return sum(self.values) / len(self.values)

    def reset(self):
        self.values.clear()


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two 2D points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp a value between min_val and max_val."""
    return max(min_val, min(max_val, value))


def get_font(size=24):
    """Load a clean system sans-serif font, falling back to PIL default."""
    from PIL import ImageFont

    paths = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def get_emoji_font(size=160):
    """Load a color emoji font, snapping to a valid bitmap size on macOS."""
    from PIL import ImageFont

    apple_path = "/System/Library/Fonts/Apple Color Emoji.ttc"
    if os.path.exists(apple_path):
        valid_sizes = [160, 96, 64, 48, 40, 32, 20]
        chosen = min(valid_sizes, key=lambda s: abs(s - size))
        try:
            return ImageFont.truetype(apple_path, chosen)
        except Exception:
            pass

    other_paths = [
        "C:\\Windows\\Fonts\\seguiemj.ttf",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    ]
    for p in other_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return None
