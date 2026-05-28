# ─────────────────────────────────────────────
#  config.py  —  Parámetros configurables
# ─────────────────────────────────────────────

# ── Umbrales EAR (Eye Aspect Ratio) ──────────
EAR_THRESHOLD = 0.22        # Por debajo de este valor el ojo se considera cerrado
EAR_CONSEC_FRAMES = 20      # Frames consecutivos para disparar alerta de somnolencia

# ── Umbrales MAR (Mouth Aspect Ratio) ────────
MAR_THRESHOLD = 0.60        # Por encima de este valor se detecta bostezo
MAR_CONSEC_FRAMES = 15      # Frames consecutivos para disparar alerta de bostezo

# ── Cámara / Video ───────────────────────────
CAMERA_INDEX = 0            # 0 = webcam principal
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Alertas ──────────────────────────────────
ALERT_SOUND_PATH = "assets/alert.wav"
ALERT_COOLDOWN_SEC = 3      # Segundos mínimos entre alertas sonoras

# ── Colores BGR ──────────────────────────────
COLOR_OK      = (0, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_DANGER  = (0, 0, 255)
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)

# ── Landmarks MediaPipe (índices) ─────────────
# Ojos
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Boca (para MAR)
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]