# ─────────────────────────────────────────────
#  detector.py  —  Lógica central de detección
# ─────────────────────────────────────────────

import cv2
import mediapipe as mp

import config
from utils import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    draw_eye_contour,
    draw_mouth_contour,
    draw_metric_bar,
    draw_overlay,
)


class DrowsinessDetector:
    """
    Detecta somnolencia y bostezos usando landmarks faciales de MediaPipe.

    Métricas:
      - EAR (Eye Aspect Ratio)  → ojos cerrados
      - MAR (Mouth Aspect Ratio) → bostezos
    """

    def __init__(self):
        # ── MediaPipe Face Mesh ───────────────────
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── Contadores de frames ──────────────────
        self.ear_counter  = 0   # Frames consecutivos con ojos cerrados
        self.mar_counter  = 0   # Frames consecutivos con boca abierta

        # ── Flags de alerta ───────────────────────
        self.drowsy_alert = False
        self.yawn_alert   = False

        # ── Estadísticas de sesión ────────────────
        self.total_drowsy_events = 0
        self.total_yawn_events   = 0
        self.frames_processed    = 0

    # ─────────────────────────────────────────────
    def process_frame(self, frame):
        """
        Procesa un frame y retorna el frame anotado + estado del conductor.

        Args:
            frame: imagen BGR de OpenCV

        Returns:
            annotated_frame : frame con anotaciones visuales
            status          : dict con métricas y alertas
        """
        self.frames_processed += 1
        h, w = frame.shape[:2]

        # Estado por defecto
        status = {
            "face_detected": False,
            "ear": 0.0,
            "mar": 0.0,
            "drowsy": False,
            "yawning": False,
            "ear_counter": self.ear_counter,
            "mar_counter": self.mar_counter,
        }

        # Convertir a RGB para MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            draw_overlay(frame, "Sin rostro detectado", (10, 30),
                         config.COLOR_WARNING)
            return frame, status

        # ── Landmarks del primer rostro ───────────
        lm = results.multi_face_landmarks[0].landmark
        status["face_detected"] = True

        # ── Calcular EAR ──────────────────────────
        ear_left  = eye_aspect_ratio(lm, config.LEFT_EYE,  w, h)
        ear_right = eye_aspect_ratio(lm, config.RIGHT_EYE, w, h)
        ear = (ear_left + ear_right) / 2.0
        status["ear"] = ear

        # ── Calcular MAR ──────────────────────────
        mar = mouth_aspect_ratio(lm, config.MOUTH, w, h)
        status["mar"] = mar

        # ── Lógica de contadores ──────────────────
        # EAR
        if ear < config.EAR_THRESHOLD:
            self.ear_counter += 1
        else:
            self.ear_counter = 0
            self.drowsy_alert = False

        if self.ear_counter >= config.EAR_CONSEC_FRAMES:
            if not self.drowsy_alert:
                self.total_drowsy_events += 1
                self.drowsy_alert = True
            status["drowsy"] = True

        # MAR
        if mar > config.MAR_THRESHOLD:
            self.mar_counter += 1
        else:
            self.mar_counter = 0
            self.yawn_alert = False

        if self.mar_counter >= config.MAR_CONSEC_FRAMES:
            if not self.yawn_alert:
                self.total_yawn_events += 1
                self.yawn_alert = True
            status["yawning"] = True

        status["ear_counter"] = self.ear_counter
        status["mar_counter"] = self.mar_counter

        # ── Dibujar contornos ─────────────────────
        eye_color = config.COLOR_DANGER if status["drowsy"] else config.COLOR_OK
        mouth_color = config.COLOR_WARNING if status["yawning"] else config.COLOR_OK

        draw_eye_contour(frame, lm, config.LEFT_EYE,  w, h, eye_color)
        draw_eye_contour(frame, lm, config.RIGHT_EYE, w, h, eye_color)
        draw_mouth_contour(frame, lm, config.MOUTH,   w, h, mouth_color)

        # ── Barras de métricas ────────────────────
        draw_metric_bar(frame, ear, config.EAR_THRESHOLD, "EAR", 10, 50)
        draw_metric_bar(frame, mar, config.MAR_THRESHOLD, "MAR", 10, 85)

        # ── Panel de estadísticas ─────────────────
        self._draw_stats_panel(frame, status)

        # ── Alertas visuales ──────────────────────
        if status["drowsy"]:
            self._draw_alert_banner(frame, "SOMNOLENCIA DETECTADA!", config.COLOR_DANGER)
        elif status["yawning"]:
            self._draw_alert_banner(frame, "BOSTEZO DETECTADO", config.COLOR_WARNING)

        return frame, status

    # ─────────────────────────────────────────────
    def _draw_stats_panel(self, frame, status):
        """Panel semitransparente con estadísticas en tiempo real."""
        h, w = frame.shape[:2]
        panel_x, panel_y = w - 220, 10
        panel_w, panel_h = 210, 110

        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        lines = [
            f"Eventos somnolencia: {self.total_drowsy_events}",
            f"Bostezos: {self.total_yawn_events}",
            f"Frames: {self.frames_processed}",
            f"EAR frames: {self.ear_counter}/{config.EAR_CONSEC_FRAMES}",
            f"MAR frames: {self.mar_counter}/{config.MAR_CONSEC_FRAMES}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line,
                        (panel_x + 6, panel_y + 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        config.COLOR_WHITE, 1)

    def _draw_alert_banner(self, frame, message, color):
        """Banner de alerta en la parte superior del frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, message, (text_x, 33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, config.COLOR_WHITE, 2)

    def get_session_summary(self):
        """Retorna un resumen de la sesión de detección."""
        return {
            "frames_procesados": self.frames_processed,
            "eventos_somnolencia": self.total_drowsy_events,
            "bostezos": self.total_yawn_events,
        }

    def release(self):
        """Libera recursos de MediaPipe."""
        self.face_mesh.close()
