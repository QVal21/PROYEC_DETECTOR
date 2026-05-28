# ─────────────────────────────────────────────
#  utils.py  —  Funciones auxiliares
# ─────────────────────────────────────────────

import numpy as np
import cv2
from scipy.spatial import distance as dist


def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    pts = [
        (int(landmarks[i].x * image_w), int(landmarks[i].y * image_h))
        for i in eye_indices
    ]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmarks, mouth_indices, image_w, image_h):
    pts = [
        (int(landmarks[i].x * image_w), int(landmarks[i].y * image_h))
        for i in mouth_indices
    ]
    horizontal = dist.euclidean(pts[0], pts[1])
    v1 = dist.euclidean(pts[2], pts[6])
    v2 = dist.euclidean(pts[3], pts[5])
    v3 = dist.euclidean(pts[4], pts[7])
    return (v1 + v2 + v3) / (3.0 * horizontal)


def _scale(frame, base=640):
    """Factor de escala relativo a un frame de referencia de 640px de ancho."""
    return frame.shape[1] / base


def draw_eye_contour(frame, landmarks, eye_indices, image_w, image_h, color):
    """Dibuja el contorno del ojo sobre el frame."""
    pts = np.array([
        (int(landmarks[i].x * image_w), int(landmarks[i].y * image_h))
        for i in eye_indices
    ], dtype=np.int32)
    thickness = max(1, int(_scale(frame) * 1.5))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def draw_mouth_contour(frame, landmarks, mouth_indices, image_w, image_h, color):
    """Dibuja el contorno de la boca sobre el frame."""
    pts = np.array([
        (int(landmarks[i].x * image_w), int(landmarks[i].y * image_h))
        for i in mouth_indices
    ], dtype=np.int32)
    thickness = max(1, int(_scale(frame) * 1.5))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def draw_overlay(frame, text, position, color, font_scale=None, thickness=None):
    """Dibuja texto con sombra escalado al tamaño del frame."""
    s = _scale(frame)
    fs = font_scale if font_scale is not None else max(0.5, s * 0.7)
    th = thickness if thickness is not None else max(1, int(s * 2))
    x, y = position
    cv2.putText(frame, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)


def draw_metric_bar(frame, value, threshold, label, x, y, w=None, h=None):
    """
    Dibuja una barra de progreso escalada al tamaño del frame.
    Verde = seguro, Rojo = peligro.
    """
    s = _scale(frame)
    bar_w = w if w is not None else int(200 * s)
    bar_h = h if h is not None else max(10, int(18 * s))
    font  = max(0.4, s * 0.55)
    ratio = min(value / (threshold * 2), 1.0)
    color = (0, 200, 0) if value >= threshold else (0, 0, 220)

    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(bar_w * ratio), y + bar_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (200, 200, 200), 1)
    cv2.putText(frame, f"{label}: {value:.2f}",
                (x, y - max(4, int(6 * s))),
                cv2.FONT_HERSHEY_SIMPLEX, font, (220, 220, 220),
                max(1, int(s)))