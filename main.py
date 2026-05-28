# ─────────────────────────────────────────────
#  main.py  —  Punto de entrada principal
# ─────────────────────────────────────────────
#
#  Uso:
#    python main.py                    → webcam
#    python main.py --source video.mp4 → archivo de video
#    python main.py --source 0         → cámara índice 0
#    python main.py --no-sound         → sin alertas sonoras
#    python main.py --save output.avi  → guardar video procesado
#
# ─────────────────────────────────────────────

import argparse
import sys
import cv2

import config
from detector import DrowsinessDetector
from alerter import Alerter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detector de somnolencia para conductores"
    )
    parser.add_argument(
        "--source",
        default=str(config.CAMERA_INDEX),
        help="Fuente de video: índice de cámara (0,1,...) o ruta a archivo de video"
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Desactivar alertas sonoras"
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Ruta para guardar el video procesado (ej: output.avi)"
    )
    return parser.parse_args()


def open_source(source_str):
    """
    Abre la fuente de video (cámara o archivo).
    Retorna el objeto VideoCapture o None si falla.
    """
    # Si es un número, tratarlo como índice de cámara
    try:
        source = int(source_str)
    except ValueError:
        source = source_str  # Es una ruta de archivo

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente: {source}")
        return None

    # Configurar resolución si es cámara
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    return cap


def main():
    args = parse_args()

    print("=" * 50)
    print("  DETECTOR DE SOMNOLENCIA PARA CONDUCTORES")
    print("=" * 50)
    print(f"  Fuente  : {args.source}")
    print(f"  Sonido  : {'Desactivado' if args.no_sound else 'Activado'}")
    print(f"  Guardar : {args.save or 'No'}")
    print("=" * 50)
    print("  Presiona 'q' para salir")
    print("  Presiona 'r' para resetear contadores")
    print("=" * 50)

    # ── Inicializar componentes ───────────────
    cap = open_source(args.source)
    if cap is None:
        sys.exit(1)

    detector = DrowsinessDetector()
    alerter  = Alerter()

    # Generar beep si no existe el archivo de audio
    if not args.no_sound:
        import os
        if not os.path.exists(config.ALERT_SOUND_PATH):
            alerter.generate_beep_wav(config.ALERT_SOUND_PATH)
            alerter._init_audio()  # Reiniciar con el archivo recién creado

    # ── Configurar grabación (opcional) ───────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        print(f"[INFO] Grabando en: {args.save}")

    # ── Bucle principal ───────────────────────
    try:
        cv2.namedWindow("Detector de Somnolencia", cv2.WINDOW_NORMAL)  # ← agrega esto
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fin del video o error de captura.")
                break

            # Procesar frame
            annotated, status = detector.process_frame(frame)

            # Disparar alertas
            if not args.no_sound:
                if status["drowsy"]:
                    alerter.trigger("drowsy")
                elif status["yawning"]:
                    alerter.trigger("yawn")

            # Mostrar FPS
            fps_text = f"FPS: {cap.get(cv2.CAP_PROP_FPS):.0f}"
            cv2.putText(annotated, fps_text,
                        (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        config.COLOR_WHITE, 1)

            # Mostrar frame
            cv2.imshow("Detector de Somnolencia", annotated)

            cv2.imshow("Detector de Somnolencia", annotated)
            cv2.resizeWindow("Detector de Somnolencia", config.FRAME_WIDTH, config.FRAME_HEIGHT)  # ← agrega esto

            # Guardar si aplica
            if writer:
                writer.write(annotated)

            # ── Teclas ────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] Saliendo...")
                break
            elif key == ord("r"):
                detector.ear_counter  = 0
                detector.mar_counter  = 0
                detector.drowsy_alert = False
                detector.yawn_alert   = False
                print("[INFO] Contadores reseteados.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario.")

    finally:
        # ── Resumen de sesión ─────────────────
        summary = detector.get_session_summary()
        print("\n" + "=" * 40)
        print("  RESUMEN DE SESIÓN")
        print("=" * 40)
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")
        print("=" * 40)

        # ── Liberar recursos ──────────────────
        cap.release()
        if writer:
            writer.release()
        detector.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
