# ─────────────────────────────────────────────
#  config.py  —  Parámetros configurables
# ─────────────────────────────────────────────

# ── Umbrales EAR (Eye Aspect Ratio) ──────────
EAR_THRESHOLD = 0.22        # Por debajo de este valor el ojo se considera cerrado
EAR_CONSEC_FRAMES = 20      # Frames consecutivos para disparar NIVEL 1
EAR_SECONDS_LEVEL2 = 2.0    # Segundos con ojos cerrados para disparar NIVEL 2

# ── Umbrales MAR (Mouth Aspect Ratio) ────────
MAR_THRESHOLD = 0.60
MAR_CONSEC_FRAMES = 15

# ── Cámara / Video ───────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Alertas ──────────────────────────────────
ALERT_SOUND_LEVEL1 = "assets/alert_level1.wav"   # Pitido corto
ALERT_SOUND_LEVEL2 = "assets/alert_level2.wav"   # Sirena
ALERT_SOUND_YAWN   = "assets/alert_yawn.wav"     # Bostezo
ALERT_COOLDOWN_L1  = 2      # Cooldown nivel 1 (segundos)
ALERT_COOLDOWN_L2  = 1      # Cooldown nivel 2 (más frecuente)

# ── Colores BGR ──────────────────────────────
COLOR_OK       = (0, 200, 0)
COLOR_WARNING  = (0, 165, 255)
COLOR_DANGER   = (0, 0, 255)
COLOR_CRITICAL = (0, 0, 180)
COLOR_WHITE    = (255, 255, 255)
COLOR_BLACK    = (0, 0, 0)

# ── Landmarks MediaPipe (índices) ─────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]
