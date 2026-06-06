# ─────────────────────────────────────────────
#  alerter.py  —  Sistema de alertas (2 niveles)
# ─────────────────────────────────────────────

import time
import threading
import os
import numpy as np
import wave
import config


class Alerter:
    """
    Nivel 1 → pitido corto y agudo (ojos cerrados por N frames)
    Nivel 2 → sirena ascendente (ojos cerrados por 2 segundos)
    """

    def __init__(self):
        self._last_l1 = 0
        self._last_l2 = 0
        self._last_yawn = 0
        self._pygame_ok = False
        self._lock = threading.Lock()

        # Generar sonidos si no existen
        os.makedirs("assets", exist_ok=True)
        if not os.path.exists(config.ALERT_SOUND_LEVEL1):
            self._gen_beep(config.ALERT_SOUND_LEVEL1, freq=1200, duration=0.4, style="beep")
        if not os.path.exists(config.ALERT_SOUND_LEVEL2):
            self._gen_beep(config.ALERT_SOUND_LEVEL2, freq=800,  duration=1.2, style="siren")
        if not os.path.exists(config.ALERT_SOUND_YAWN):
            self._gen_beep(config.ALERT_SOUND_YAWN,   freq=600,  duration=0.3, style="beep")

        self._init_audio()

    # ── Init pygame ───────────────────────────
    def _init_audio(self):
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._sounds = {
                "level1": pygame.mixer.Sound(config.ALERT_SOUND_LEVEL1),
                "level2": pygame.mixer.Sound(config.ALERT_SOUND_LEVEL2),
                "yawn":   pygame.mixer.Sound(config.ALERT_SOUND_YAWN),
            }
            # Canales separados para que nivel2 no sea interrumpido por nivel1
            self._ch_l1   = pygame.mixer.Channel(0)
            self._ch_l2   = pygame.mixer.Channel(1)
            self._ch_yawn = pygame.mixer.Channel(2)
            self._pygame_ok = True
            print("[Alerter] Audio iniciado correctamente.")
        except Exception as e:
            print(f"[Alerter] Sin audio: {e}")

    # ── Trigger público ───────────────────────
    def trigger(self, alert_type: str):
        """
        alert_type: "level1" | "level2" | "yawn"
        """
        now = time.time()
        with self._lock:
            if alert_type == "level1":
                if now - self._last_l1 < config.ALERT_COOLDOWN_L1:
                    return
                self._last_l1 = now
            elif alert_type == "level2":
                if now - self._last_l2 < config.ALERT_COOLDOWN_L2:
                    return
                self._last_l2 = now
            elif alert_type == "yawn":
                if now - self._last_yawn < 3:
                    return
                self._last_yawn = now

        labels = {"level1": "NIVEL 1 — Pitido", "level2": "NIVEL 2 — SIRENA", "yawn": "Bostezo"}
        print(f"\n[ALERTA] {labels.get(alert_type, alert_type)} — {time.strftime('%H:%M:%S')}")

        if self._pygame_ok:
            threading.Thread(target=self._play, args=(alert_type,), daemon=True).start()

    def stop_level2(self):
        """Detiene la sirena del nivel 2 inmediatamente."""
        if self._pygame_ok:
            try:
                self._ch_l2.stop()
            except Exception:
                pass

    # ── Reproducción ──────────────────────────
    def _play(self, alert_type):
        try:
            if alert_type == "level1":
                self._ch_l1.play(self._sounds["level1"])
            elif alert_type == "level2":
                self._ch_l2.play(self._sounds["level2"], loops=0)
            elif alert_type == "yawn":
                self._ch_yawn.play(self._sounds["yawn"])
        except Exception as e:
            print(f"[Alerter] Error reproduciendo {alert_type}: {e}")

    # ── Generador de WAV ──────────────────────
    def _gen_beep(self, path, freq=880, duration=0.5, style="beep"):
        """
        Genera archivos WAV sintéticos.
        style="beep"  → onda senoidal con fade out (pitido)
        style="siren" → frecuencia que sube y baja (sirena)
        """
        sr = 44100
        n  = int(sr * duration)
        t  = np.linspace(0, duration, n, False)

        if style == "siren":
            # Frecuencia oscila entre freq y freq*2
            mod = freq + freq * np.abs(np.sin(2 * np.pi * 2.5 * t))
            wave_data = np.sin(2 * np.pi * np.cumsum(mod) / sr)
            # Envelope: sube rápido, mantiene, baja
            env = np.ones(n)
            ramp = int(sr * 0.05)
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        else:
            wave_data = np.sin(2 * np.pi * freq * t)
            env = np.linspace(1.0, 0.0, n)

        samples = (wave_data * env * 32767).astype(np.int16)

        with wave.open(path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())
        print(f"[Alerter] Generado: {path}")
