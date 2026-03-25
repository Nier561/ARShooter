"""
debug_coords.py
Muestra en tiempo real los valores RAW de los landmarks para entender
exactamente el sistema de coordenadas antes de corregir el juego.
Mueve el dedo índice y observa cómo cambian los valores.
"""
import cv2
import numpy as np
import mediapipe as mp
import os, tempfile, time

# ── Cargar modelo ──────────────────────────────────────────────────────────
model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
from mediapipe.tasks import python as _mpt
from mediapipe.tasks.python import vision as _mpv

base_opts = _mpt.BaseOptions(model_asset_path=model_path)
opts = _mpv.HandLandmarkerOptions(
    base_options=base_opts, num_hands=2,
    running_mode=_mpv.RunningMode.VIDEO,
)
detector = _mpv.HandLandmarker.create_from_options(opts)

# ── Abrir cámara ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

INDEX_MCP, INDEX_TIP = 5, 8

print("Instrucciones:")
print("  - Pon la mano DERECHA frente a la cámara")
print("  - Apunta el dedo índice a distintas direcciones")
print("  - Observa los valores en consola y en la ventana")
print("  - Q para salir")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Probar CON y SIN flip para ver cuál es correcto
    flipped = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)

    ts_ms = int(time.monotonic() * 1000)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_img, ts_ms)

    display = flipped.copy()
    H, W = display.shape[:2]

    for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
        label_raw = handedness[0].category_name  # lo que dice MediaPipe
        lm = [(p.x, p.y, p.z) for p in hand_lm]

        mcp = lm[INDEX_MCP]
        tip = lm[INDEX_TIP]

        # Delta RAW (sin ninguna inversión)
        dx_raw = tip[0] - mcp[0]
        dy_raw = tip[1] - mcp[1]

        # Puntos en píxeles
        mcp_px = (int(mcp[0]*W), int(mcp[1]*H))
        tip_px = (int(tip[0]*W), int(tip[1]*H))

        # Dibujar flecha de la dirección RAW
        cv2.arrowedLine(display, mcp_px, tip_px, (0,255,0), 3)
        cv2.circle(display, tip_px, 8, (0,0,255), -1)
        cv2.circle(display, mcp_px, 6, (255,255,0), -1)

        # Info en pantalla
        info = f"Label={label_raw}  dx={dx_raw:+.3f}  dy={dy_raw:+.3f}"
        cv2.putText(display, info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Indicador de dirección
        dir_x = "→ DER imagen" if dx_raw > 0 else "← IZQ imagen"
        dir_y = "↓ ABAJO imagen" if dy_raw > 0 else "↑ ARRIBA imagen"
        cv2.putText(display, f"X: {dir_x}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
        cv2.putText(display, f"Y: {dir_y}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

        # Imprimir en consola también
        print(f"\rLabel={label_raw:5s}  tip=({tip[0]:.3f},{tip[1]:.3f})  "
              f"mcp=({mcp[0]:.3f},{mcp[1]:.3f})  "
              f"dx={dx_raw:+.3f}  dy={dy_raw:+.3f}  ", end="", flush=True)

    cv2.imshow("DEBUG - Coordenadas RAW (imagen espejada)", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\nListo.")
