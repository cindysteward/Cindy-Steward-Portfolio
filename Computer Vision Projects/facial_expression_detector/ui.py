"""
UI rendering for the facial expression mimicry application.
Draws the full display layout including emoji expression prompts, camera feed with
landmark overlay, progress bar, status messages, and the well-done celebration overlay.
"""

import cv2
import numpy as np
from typing import List, Optional
from PIL import Image, ImageDraw

from utils import (
    get_font, get_emoji_font, EXPRESSION_EMOJIS,
    DISPLAY_WIDTH, DISPLAY_HEIGHT, LEFT_PANEL_WIDTH,
    TITLE_BAR_HEIGHT, BOTTOM_BAR_HEIGHT,
    TITLE_TEXT, NO_FACE_MESSAGE,
    TITLE_FONT_SIZE, EXPRESSION_NAME_FONT_SIZE, STATUS_FONT_SIZE,
    WELLDONE_FONT_SIZE, NO_FACE_FONT_SIZE, HINT_FONT_SIZE, COUNTER_FONT_SIZE,
    COLOR_BG, COLOR_PANEL_BG, COLOR_TITLE_BAR, COLOR_ACCENT, COLOR_SUCCESS,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED,
    COLOR_BAR_BG,
    BGR_TITLE_BAR, BGR_BG, BGR_PANEL_BG, BGR_DIVIDER,
    BGR_LANDMARK_EYE, BGR_LANDMARK_BROW, BGR_LANDMARK_MOUTH, BGR_LANDMARK_CONTOUR,
)
from face_mesh import FaceLandmarks


def create_display_window(name: str):
    """Create and size the named OpenCV display window."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, DISPLAY_WIDTH, DISPLAY_HEIGHT)


def draw_facial_landmarks(frame, landmarks, frame_width, frame_height):
    """Draw face mesh landmarks (contour, eyes, brows, mouth) on the camera frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    if len(landmarks) > 200:
        contour = []
        for idx in FaceLandmarks.FACE_CONTOUR:
            x = int(landmarks[idx][0] * w)
            y = int(landmarks[idx][1] * h)
            contour.append([x, y])
        cv2.polylines(out, [np.array(contour)], True, BGR_LANDMARK_CONTOUR, 1, cv2.LINE_AA)

    for eye_indices in [FaceLandmarks.LEFT_EYE, FaceLandmarks.RIGHT_EYE]:
        for idx in eye_indices:
            x = int(landmarks[idx][0] * w)
            y = int(landmarks[idx][1] * h)
            cv2.circle(out, (x, y), 2, BGR_LANDMARK_EYE, -1, cv2.LINE_AA)

    for brow_indices in [FaceLandmarks.LEFT_EYEBROW, FaceLandmarks.RIGHT_EYEBROW]:
        pts = []
        for idx in brow_indices:
            x = int(landmarks[idx][0] * w)
            y = int(landmarks[idx][1] * h)
            pts.append([x, y])
        cv2.polylines(out, [np.array(pts)], False, BGR_LANDMARK_BROW, 2, cv2.LINE_AA)

    mouth_pts = []
    for idx in FaceLandmarks.MOUTH:
        x = int(landmarks[idx][0] * w)
        y = int(landmarks[idx][1] * h)
        mouth_pts.append([x, y])
    cv2.polylines(out, [np.array(mouth_pts)], True, BGR_LANDMARK_MOUTH, 1, cv2.LINE_AA)

    return out


def assemble_frame(
    camera_frame: Optional[np.ndarray],
    face_detected: bool,
    landmarks: Optional[List],
    expression_name: str,
    progress_percent: float,
    show_well_done: bool,
    calibrating: bool,
    calibration_progress: float,
    expression_index: int,
    total_expressions: int,
) -> np.ndarray:
    """Compose the full display frame with all UI panels and overlays."""
    W, H = DISPLAY_WIDTH, DISPLAY_HEIGHT
    LP = LEFT_PANEL_WIDTH
    TB = TITLE_BAR_HEIGHT
    BB = BOTTOM_BAR_HEIGHT
    cam_panel_w = W - LP
    cam_area_h = H - TB - BB

    frame = np.full((H, W, 3), 240, dtype=np.uint8)
    frame[:TB, :] = BGR_TITLE_BAR
    frame[TB:H - BB, :LP] = BGR_PANEL_BG
    frame[H - BB:, :] = BGR_PANEL_BG
    frame[TB:H - BB, LP:] = BGR_BG

    cv2.line(frame, (LP, TB), (LP, H - BB), BGR_DIVIDER, 1)
    cv2.line(frame, (0, H - BB), (W, H - BB), BGR_DIVIDER, 1)

    illu_size = min(LP - 40, cam_area_h - 130)
    illu_size = max(illu_size, 60)
    illu_y = TB + 55

    cam_margin = 12
    cam_area_x = LP + cam_margin
    cam_area_y = TB + cam_margin
    cam_fit_w = cam_panel_w - cam_margin * 2
    cam_fit_h = cam_area_h - cam_margin * 2

    if camera_frame is not None:
        cam = camera_frame.copy()
        if face_detected and landmarks:
            cam = draw_facial_landmarks(cam, landmarks, cam.shape[1], cam.shape[0])
        ch, cw = cam.shape[:2]
        scale = min(cam_fit_w / cw, cam_fit_h / ch)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        cam_resized = cv2.resize(cam, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x_off = cam_area_x + (cam_fit_w - new_w) // 2
        y_off = cam_area_y + (cam_fit_h - new_h) // 2
        frame[y_off:y_off + new_h, x_off:x_off + new_w] = cam_resized
        cv2.rectangle(frame, (x_off - 1, y_off - 1),
                      (x_off + new_w, y_off + new_h), BGR_DIVIDER, 1, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title_font = get_font(TITLE_FONT_SIZE)
    name_font = get_font(EXPRESSION_NAME_FONT_SIZE)
    status_font = get_font(STATUS_FONT_SIZE)
    hint_font = get_font(HINT_FONT_SIZE)
    counter_font = get_font(COUNTER_FONT_SIZE)
    welldone_font = get_font(WELLDONE_FONT_SIZE)
    noface_font = get_font(NO_FACE_FONT_SIZE)

    title_bbox = draw.textbbox((0, 0), TITLE_TEXT, font=title_font)
    title_tw = title_bbox[2] - title_bbox[0]
    title_th = title_bbox[3] - title_bbox[1]
    title_x = (W - title_tw) // 2
    title_y = (TB - title_th) // 2
    draw.text((title_x, title_y), TITLE_TEXT, font=title_font, fill=(255, 255, 255))

    quit_text = "Q / ESC quit  ·  R recalibrate  ·  S skip"
    quit_bbox = draw.textbbox((0, 0), quit_text, font=hint_font)
    quit_x = W - (quit_bbox[2] - quit_bbox[0]) - 16
    quit_y = (TB - (quit_bbox[3] - quit_bbox[1])) // 2
    draw.text((quit_x, quit_y), quit_text, font=hint_font, fill=(160, 175, 190))

    if not calibrating:
        name_text = expression_name
    else:
        name_text = "Get Ready"
    name_bbox = draw.textbbox((0, 0), name_text, font=name_font)
    name_w = name_bbox[2] - name_bbox[0]
    name_x = (LP - name_w) // 2
    name_y = TB + 18
    draw.text((name_x, name_y), name_text, font=name_font, fill=COLOR_TEXT_PRIMARY)

    emoji_font = get_emoji_font(160)
    emoji = EXPRESSION_EMOJIS.get(expression_name, "\U0001f610") if not calibrating else "\U0001f610"
    if emoji_font is not None:
        emoji_img = Image.new("RGBA", (512, 512), (255, 255, 255, 0))
        emoji_draw = ImageDraw.Draw(emoji_img)
        emoji_draw.text((0, 0), emoji, font=emoji_font, embedded_color=True)
        bbox = emoji_img.getbbox()
        if bbox:
            emoji_img = emoji_img.crop(bbox)
        target_size = int(illu_size * 0.75)
        emoji_img = emoji_img.resize((target_size, target_size), Image.LANCZOS)
        paste_x = (LP - target_size) // 2
        paste_y = illu_y + (illu_size - target_size) // 2
        rgb_pil = pil.convert("RGBA")
        rgb_pil.paste(emoji_img, (paste_x, paste_y), emoji_img)
        pil = rgb_pil.convert("RGB")
        draw = ImageDraw.Draw(pil)
    else:
        fallback = EXPRESSION_EMOJIS.get(expression_name, "?") if not calibrating else "?"
        fb_font = get_font(int(illu_size * 0.5))
        fb_bbox = draw.textbbox((0, 0), fallback, font=fb_font)
        fb_w = fb_bbox[2] - fb_bbox[0]
        fb_h = fb_bbox[3] - fb_bbox[1]
        draw.text(((LP - fb_w) // 2, illu_y + (illu_size - fb_h) // 2),
                  fallback, font=fb_font, fill=COLOR_TEXT_PRIMARY)

    if not calibrating:
        counter_text = f"{expression_index} / {total_expressions}"
        counter_bbox = draw.textbbox((0, 0), counter_text, font=counter_font)
        counter_x = (LP - (counter_bbox[2] - counter_bbox[0])) // 2
        draw.text((counter_x, illu_y + illu_size + 10), counter_text, font=counter_font,
                  fill=COLOR_TEXT_SECONDARY)

    if calibrating:
        cal_lines = ["Look at the camera", "with a relaxed face\u2026"]
        cal_line_y = illu_y + illu_size + 15
        for line in cal_lines:
            lb = draw.textbbox((0, 0), line, font=hint_font)
            lw = lb[2] - lb[0]
            lh = lb[3] - lb[1]
            lx = (LP - lw) // 2
            draw.text((lx, cal_line_y), line, font=hint_font, fill=COLOR_TEXT_SECONDARY)
            cal_line_y += lh + 4

    if not calibrating and not face_detected:
        nf_bbox = draw.textbbox((0, 0), NO_FACE_MESSAGE, font=noface_font)
        nf_tw = nf_bbox[2] - nf_bbox[0]
        nf_x = LP + (cam_panel_w - nf_tw) // 2
        nf_y = TB + cam_area_h // 2 - 10
        draw.text((nf_x, nf_y), NO_FACE_MESSAGE, font=noface_font,
                  fill=COLOR_TEXT_SECONDARY)

    bar_margin_x = 35
    bar_x = bar_margin_x
    bar_w = W - bar_margin_x * 2 - 130
    bar_h = 22
    bar_y = H - BB + (BB - bar_h) // 2
    bar_radius = bar_h // 2

    draw.rounded_rectangle(
        [(bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h)],
        radius=bar_radius, fill=COLOR_BAR_BG,
    )

    if calibrating:
        fill_frac = calibration_progress
        fill_color = COLOR_ACCENT
    elif show_well_done:
        fill_frac = 1.0
        fill_color = COLOR_SUCCESS
    else:
        fill_frac = progress_percent / 100.0
        fill_color = COLOR_ACCENT

    fill_w = int(bar_w * fill_frac)
    if fill_w > 0:
        fill_w = max(fill_w, bar_h)
        fill_w = min(fill_w, bar_w)
        draw.rounded_rectangle(
            [(bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h)],
            radius=bar_radius, fill=fill_color,
        )

    if calibrating:
        status_text = f"{int(calibration_progress * 100)}%  Calibrating\u2026"
        status_color = COLOR_TEXT_SECONDARY
    elif show_well_done:
        status_text = "100%  \u2714 Done!"
        status_color = COLOR_SUCCESS
    elif progress_percent > 0:
        status_text = f"{int(progress_percent)}%  Hold it!"
        status_color = COLOR_ACCENT
    else:
        status_text = "Waiting\u2026"
        status_color = COLOR_TEXT_MUTED

    status_x = bar_x + bar_w + 14
    status_bbox = draw.textbbox((0, 0), status_text, font=status_font)
    status_th = status_bbox[3] - status_bbox[1]
    status_y = bar_y + (bar_h - status_th) // 2 - 1
    draw.text((status_x, status_y), status_text, font=status_font, fill=status_color)

    if show_well_done and not calibrating:
        wd_text = "Well Done!"
        wd_bbox = draw.textbbox((0, 0), wd_text, font=welldone_font)
        wd_w = wd_bbox[2] - wd_bbox[0]
        wd_h = wd_bbox[3] - wd_bbox[1]
        pad = 30
        rect_cx = LP + cam_panel_w // 2
        rect_cy = TB + cam_area_h // 2
        rect_x1 = rect_cx - wd_w // 2 - pad
        rect_y1 = rect_cy - wd_h // 2 - pad
        rect_x2 = rect_cx + wd_w // 2 + pad
        rect_y2 = rect_cy + wd_h // 2 + pad
        wd_x = rect_cx - (wd_bbox[0] + wd_bbox[2]) // 2
        wd_y = rect_cy - (wd_bbox[1] + wd_bbox[3]) // 2

        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            [(rect_x1, rect_y1), (rect_x2, rect_y2)],
            radius=20, fill=(39, 174, 96, 210),
        )
        pil = Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(pil)
        draw.text((wd_x, wd_y), wd_text, font=welldone_font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
