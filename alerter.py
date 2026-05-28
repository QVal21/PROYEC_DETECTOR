# ─────────────────────────────────────────────
#  alerter.py  —  Sistema de alertas
# ─────────────────────────────────────────────

import time
import threading
import os
import config


class Alerter:
    """
    Maneja alertas sonoras con cooldown para evitar spam de audio.
    Usa pygame si está disponible; si no, imprime en consola.
    """

    def __init__(self):
        self._last_alert_time = 0
        self._pygame_ok = False
        self._lock = threading.Lock()
        self._init_audio()

    def _init_audio(self):
        """Intenta inicializar pygame para el audio."""
        try:
            import pygame
            pygame.mixer.init()
            if os.path.exists(config.ALERT_SOUND_PATH):
                self._sound = pygame.mixer.Sound(config.ALERT_SOUND_PATH)
                self._pygame_ok = True
                print("[Alerter] Audio iniciado correctamente.")
            else:
                print(f"[Alerter] Archivo de audio no encontrado: {config.ALERT_SOUND_PATH}")
                print("[Alerter] Continuando sin audio.")
        except ImportError:
            print("[Alerter] pygame no instalado. Alertas solo visuales.")
        except Exception as e:
            print(f"[Alerter] Error al iniciar audio: {e}")

    def trigger(self, alert_type="drowsy"):
        """
        Dispara una alerta si ha pasado el cooldown.

        Args:
            alert_type: "drowsy" | "yawn"
        """
        now = time.time()
        with self._lock:
            if now - self._last_alert_time < config.ALERT_COOLDOWN_SEC:
                return  # Aún en cooldown
            self._last_alert_time = now

        label = "¡SOMNOLENCIA!" if alert_type == "drowsy" else "¡BOSTEZO!"
        print(f"\n[ALERTA] {label} — {time.strftime('%H:%M:%S')}")

        if self._pygame_ok:
            threading.Thread(target=self._play_sound, daemon=True).start()

    def _play_sound(self):
        """Reproduce el sonido de alerta en un hilo separado."""
        try:
            self._sound.play()
            time.sleep(self._sound.get_length())
        except Exception as e:
            print(f"[Alerter] Error reproduciendo sonido: {e}")

    def generate_beep_wav(self, path="assets/alert.wav"):
        """
        Genera un archivo WAV de beep simple usando numpy.
        Útil si no tienes un archivo de audio propio.
        """
        try:
            import numpy as np
            import wave, struct

            os.makedirs(os.path.dirname(path), exist_ok=True)

            sample_rate = 44100
            duration    = 0.8     # segundos
            frequency   = 880     # Hz (La5)
            num_samples = int(sample_rate * duration)

            t = np.linspace(0, duration, num_samples, False)
            # Onda sinusoidal con fade out
            wave_data = np.sin(2 * np.pi * frequency * t)
            fade = np.linspace(1.0, 0.0, num_samples)
            wave_data = (wave_data * fade * 32767).astype(np.int16)

            with wave.open(path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(wave_data.tobytes())

            print(f"[Alerter] Beep generado en: {path}")
            return True
        except Exception as e:
            print(f"[Alerter] No se pudo generar beep: {e}")
            return False
