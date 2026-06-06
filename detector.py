# ─────────────────────────────────────────────
#  detector.py  —  Lógica central de detección
# ─────────────────────────────────────────────

import time
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
    Nivel 1 — ojos cerrados por EAR_CONSEC_FRAMES frames  → pitido
    Nivel 2 — ojos cerrados por EAR_SECONDS_LEVEL2 segundos → sirena
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── Contadores ────────────────────────
        self.ear_counter = 0
        self.mar_counter = 0

        # ── Nivel 2: tiempo ───────────────────
        self._eyes_closed_since = None   # timestamp cuando empezaron a cerrarse
        self.level2_active = False       # sirena activa

        # ── Flags nivel 1 ─────────────────────
        self.drowsy_alert = False
        self.yawn_alert   = False

        # ── Estadísticas ──────────────────────
        self.total_l1_events   = 0
        self.total_l2_events   = 0
        self.total_yawn_events = 0
        self.frames_processed  = 0

    # ─────────────────────────────────────────
    def process_frame(self, frame):
        self.frames_processed += 1
        h, w = frame.shape[:2]

        status = {
            "face_detected": False,
            "ear": 0.0,
            "mar": 0.0,
            "alert_level": 0,   # 0=nada, 1=pitido, 2=sirena
            "yawning": False,
            "ear_counter": self.ear_counter,
            "eyes_closed_seconds": 0.0,
        }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            draw_overlay(frame, "Sin rostro detectado", (10, 30), config.COLOR_WARNING)
            self._reset_eye_timers()
            return frame, status

        lm = results.multi_face_landmarks[0].landmark
        status["face_detected"] = True

        # ── EAR ───────────────────────────────
        ear_l = eye_aspect_ratio(lm, config.LEFT_EYE,  w, h)
        ear_r = eye_aspect_ratio(lm, config.RIGHT_EYE, w, h)
        ear   = (ear_l + ear_r) / 2.0
        status["ear"] = ear

        # ── MAR ───────────────────────────────
        mar = mouth_aspect_ratio(lm, config.MOUTH, w, h)
        status["mar"] = mar

        # ── Lógica EAR ────────────────────────
        if ear < config.EAR_THRESHOLD:
            self.ear_counter += 1

            # Nivel 2: cronometrar tiempo con ojos cerrados
            if self._eyes_closed_since is None:
                self._eyes_closed_since = time.time()

            closed_secs = time.time() - self._eyes_closed_since
            status["eyes_closed_seconds"] = closed_secs

            # NIVEL 2 (prioridad)
            if closed_secs >= config.EAR_SECONDS_LEVEL2:
                if not self.level2_active:
                    self.total_l2_events += 1
                    self.level2_active = True
                status["alert_level"] = 2

            # NIVEL 1
            elif self.ear_counter >= config.EAR_CONSEC_FRAMES:
                if not self.drowsy_alert:
                    self.total_l1_events += 1
                    self.drowsy_alert = True
                status["alert_level"] = 1

        else:
            # Ojos abiertos → resetear todo
            self._reset_eye_timers()

        # ── Lógica MAR ────────────────────────
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

        # ── Dibujar ───────────────────────────
        if status["alert_level"] == 2:
            eye_color = config.COLOR_CRITICAL
        elif status["alert_level"] == 1:
            eye_color = config.COLOR_DANGER
        else:
            eye_color = config.COLOR_OK

        mouth_color = config.COLOR_WARNING if status["yawning"] else config.COLOR_OK

        draw_eye_contour(frame, lm, config.LEFT_EYE,  w, h, eye_color)
        draw_eye_contour(frame, lm, config.RIGHT_EYE, w, h, eye_color)
        draw_mouth_contour(frame, lm, config.MOUTH,   w, h, mouth_color)

        draw_metric_bar(frame, ear, config.EAR_THRESHOLD, "EAR", 10, 50)
        draw_metric_bar(frame, mar, config.MAR_THRESHOLD, "MAR", 10, 90)

        self._draw_stats_panel(frame, status)
        self._draw_alert_banner(frame, status)

        return frame, status

    # ── Reset timers ──────────────────────────
    def _reset_eye_timers(self):
        self.ear_counter = 0
        self._eyes_closed_since = None
        self.drowsy_alert = False
        if self.level2_active:
            self.level2_active = False   # señal para que main detenga sirena

    def reset_level2(self):
        """Llamado desde main cuando el usuario presiona 's'."""
        self.level2_active = False
        self._eyes_closed_since = None

    # ── Panels ────────────────────────────────
    def _draw_stats_panel(self, frame, status):
        h, w = frame.shape[:2]
        px, py, pw, ph = w - 230, 10, 220, 125
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        secs = f"{status['eyes_closed_seconds']:.1f}s" if status['eyes_closed_seconds'] > 0 else "--"
        lines = [
            f"Nivel 1 (pitido): {self.total_l1_events}",
            f"Nivel 2 (sirena): {self.total_l2_events}",
            f"Bostezos: {self.total_yawn_events}",
            f"EAR frames: {self.ear_counter}/{config.EAR_CONSEC_FRAMES}",
            f"Ojos cerrados: {secs} / {config.EAR_SECONDS_LEVEL2}s",
            f"Frames: {self.frames_processed}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (px + 6, py + 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.COLOR_WHITE, 1)

    def _draw_alert_banner(self, frame, status):
        h, w = frame.shape[:2]
        level = status["alert_level"]

        if level == 2:
            color   = config.COLOR_CRITICAL
            message = "⚠  NIVEL 2 — PELIGRO: OJOS CERRADOS 2s"
        elif level == 1:
            color   = config.COLOR_DANGER
            message = "NIVEL 1 — SOMNOLENCIA DETECTADA"
        elif status["yawning"]:
            color   = config.COLOR_WARNING
            message = "BOSTEZO DETECTADO"
        else:
            return

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        ts = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0]
        cv2.putText(frame, message, ((w - ts[0]) // 2, 33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, config.COLOR_WHITE, 2)

        # Barra de progreso nivel 2
        if level >= 1 and self._eyes_closed_since:
            secs    = time.time() - self._eyes_closed_since
            ratio   = min(secs / config.EAR_SECONDS_LEVEL2, 1.0)
            bar_w   = int(w * ratio)
            bar_col = config.COLOR_CRITICAL if ratio >= 1.0 else config.COLOR_DANGER
            cv2.rectangle(frame, (0, 48), (bar_w, 54), bar_col, -1)

    def get_session_summary(self):
        return {
            "frames_procesados":    self.frames_processed,
            "eventos_nivel1":       self.total_l1_events,
            "eventos_nivel2":       self.total_l2_events,
            "bostezos":             self.total_yawn_events,
        }

    def release(self):
        self.face_mesh.close()
