# ─────────────────────────────────────────────
#  main.py  —  Punto de entrada principal
# ─────────────────────────────────────────────
#
#  Uso:
#    python main.py                    → webcam
#    python main.py --source video.mp4 → archivo de video
#    python main.py --no-sound         → sin alertas sonoras
#    python main.py --save output.avi  → guardar video procesado
#
#  Teclas:
#    q → salir
#    r → resetear contadores
#    s → silenciar / resetear nivel 2
#
# ─────────────────────────────────────────────

import argparse
import sys
import cv2

import config
from detector import DrowsinessDetector
from alerter import Alerter


def parse_args():
    parser = argparse.ArgumentParser(description="Detector de somnolencia para conductores")
    parser.add_argument("--source", default=str(config.CAMERA_INDEX))
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--save", default=None)
    return parser.parse_args()


def open_source(source_str):
    try:
        source = int(source_str)
    except ValueError:
        source = source_str
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {source}")
        return None
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def main():
    args = parse_args()

    print("=" * 55)
    print("  DETECTOR DE SOMNOLENCIA — 2 NIVELES DE ALERTA")
    print("=" * 55)
    print(f"  Nivel 1 : pitido  ({config.EAR_CONSEC_FRAMES} frames ojos cerrados)")
    print(f"  Nivel 2 : sirena  ({config.EAR_SECONDS_LEVEL2}s ojos cerrados)")
    print(f"  Teclas  : q=salir  r=reset  s=silenciar nivel 2")
    print("=" * 55)

    cap = open_source(args.source)
    if cap is None:
        sys.exit(1)

    detector = DrowsinessDetector()
    alerter  = Alerter()

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    cv2.namedWindow("Detector de Somnolencia", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fin del video.")
                break

            annotated, status = detector.process_frame(frame)

            # ── Disparar alertas por nivel ────
            if not args.no_sound:
                level = status["alert_level"]
                if level == 2:
                    alerter.trigger("level2")
                elif level == 1:
                    alerter.trigger("level1")
                elif status["yawning"]:
                    alerter.trigger("yawn")

            cv2.imshow("Detector de Somnolencia", annotated)
            if writer:
                writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                detector.ear_counter  = 0
                detector.mar_counter  = 0
                detector.drowsy_alert = False
                detector.yawn_alert   = False
                print("[INFO] Contadores reseteados.")
            elif key == ord("s"):
                # Silenciar / resetear nivel 2
                alerter.stop_level2()
                detector.reset_level2()
                print("[INFO] Nivel 2 silenciado.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido.")
    finally:
        summary = detector.get_session_summary()
        print("\n" + "=" * 40)
        print("  RESUMEN DE SESIÓN")
        print("=" * 40)
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")
        print("=" * 40)

        cap.release()
        if writer:
            writer.release()
        detector.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
