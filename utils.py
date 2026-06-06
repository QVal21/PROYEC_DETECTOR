# ─────────────────────────────────────────────
#  utils.py  —  Funciones auxiliares
# ─────────────────────────────────────────────

import numpy as np
import cv2
from scipy.spatial import distance as dist


def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    """
    Calcula el Eye Aspect Ratio (EAR).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        landmarks   : lista de NormalizedLandmark de MediaPipe
        eye_indices : 6 índices de los landmarks del ojo
        image_w/h   : dimensiones del frame

    Returns:
        float: valor EAR (0.0 – 1.0 aprox.)
    """
    pts = [
        (int(landmarks[i].x * image_w), int(landmarks[i].y * image_h))
        for i in eye_indices
    ]

    # Distancias verticales
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    # Distancia horizontal
    C = dist.euclidean(pts[0], pts[3])

    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(landmarks, mouth_indices, image_w, image_h):
    """
    Calcula el Mouth Aspect Ratio (MAR) para detectar bostezos.

    Args:
        landmarks     : lista de NormalizedLandmark de MediaPipe
        mouth_indices : 8 índices de los landmarks de la boca
        image_w/h     : dimensiones del frame

    Returns:
        float: valor MAR
    """
    pts = [
        (int(landmarks[i].x * image_w), int(landmarks[i].y * image_h))
        for i in mouth_indices
    ]

    # Ancho de la boca (horizontal)
    horizontal = dist.euclidean(pts[0], pts[1])
    # Alto de la boca (3 medidas verticales)
    v1 = dist.euclidean(pts[2], pts[6])
    v2 = dist.euclidean(pts[3], pts[5])
    v3 = dist.euclidean(pts[4], pts[7])

    mar = (v1 + v2 + v3) / (3.0 * horizontal)
    return mar


def draw_eye_contour(frame, landmarks, eye_indices, image_w, image_h, color):
    """Dibuja el contorno del ojo sobre el frame."""
    pts = np.array([
        (int(landmarks[i].x * image_w), int(landmarks[i].y * image_h))
        for i in eye_indices
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def draw_mouth_contour(frame, landmarks, mouth_indices, image_w, image_h, color):
    """Dibuja el contorno de la boca sobre el frame."""
    pts = np.array([
        (int(landmarks[i].x * image_w), int(landmarks[i].y * image_h))
        for i in mouth_indices
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def draw_overlay(frame, text, position, color, font_scale=0.7, thickness=2):
    """Dibuja texto con sombra para mejor legibilidad."""
    x, y = position
    # Sombra
    cv2.putText(frame, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
    # Texto principal
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_metric_bar(frame, value, threshold, label, x, y, w=150, h=12):
    """
    Dibuja una barra de progreso para visualizar EAR o MAR.
    Verde = seguro, Rojo = peligro.
    """
    ratio = min(value / (threshold * 2), 1.0)
    color = (0, 200, 0) if value >= threshold else (0, 0, 220)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * ratio), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
