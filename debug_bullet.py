"""
debug_bullet.py
Muestra en tiempo real:
1. Los landmarks del índice (MCP y TIP) en valores RAW
2. El vector dirección que se calcularía para la bala
3. Una flecha visual en la ventana de cámara

Ejecuta esto y apunta el dedo en distintas direcciones.
Dime qué ves — si la flecha en pantalla coincide con donde apunta el dedo.
"""
import cv2
import numpy as np
import mediapipe as mp
import os, tempfile, time, math

model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
from mediapipe.tasks import python as _mpt
from mediapipe.tasks.python import vision as _mpv

opts = _mpv.HandLandmarkerOptions(
    base_options=_mpt.BaseOptions(model_asset_path=model_path),
    num_hands=1,
    running_mode=_mpv.RunningMode.VIDEO,
)
detector = _mpv.HandLandmarker.create_from_options(opts)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

INDEX_MCP, INDEX_TIP = 5, 8

print("Apunta el dedo índice en distintas direcciones.")
print("Observa la flecha amarilla — ¿sigue la dirección de tu dedo?")
print("Q = salir\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    flipped = cv2.flip(frame, 1)
    rgb = np.ascontiguousarray(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
    ts  = int(time.monotonic() * 1000)
    res = detector.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    display = flipped.copy()
    H, W = display.shape[:2]

    for hand_lm, handedness in zip(res.hand_landmarks, res.handedness):
        lm = [(p.x, p.y, p.z) for p in hand_lm]

        mcp = lm[INDEX_MCP]
        tip = lm[INDEX_TIP]

        mcp_px = (int(mcp[0]*W), int(mcp[1]*H))
        tip_px = (int(tip[0]*W), int(tip[1]*H))

        # Vector 2D puro en imagen
        dx2d = tip[0] - mcp[0]
        dy2d = tip[1] - mcp[1]

        # Vector 3D con Z de MediaPipe
        dx3d = tip[0] - mcp[0]
        dy3d = tip[1] - mcp[1]
        dz3d = tip[2] - mcp[2]   # Z MediaPipe: negativo = lejos de cámara

        mag2d = math.sqrt(dx2d**2 + dy2d**2)
        mag3d = math.sqrt(dx3d**2 + dy3d**2 + dz3d**2)

        # Dibujar landmarks
        cv2.circle(display, mcp_px, 8, (255,255,0), -1)
        cv2.circle(display, tip_px, 8, (0,0,255), -1)
        cv2.line(display, mcp_px, tip_px, (0,255,0), 2)

        # Flecha amplificada desde el centro de la pantalla
        cx, cy = W//2, H//2
        AMP = 200
        if mag2d > 0.001:
            arrow_end = (
                int(cx + (dx2d/mag2d) * AMP),
                int(cy + (dy2d/mag2d) * AMP),
            )
            cv2.arrowedLine(display, (cx,cy), arrow_end, (0,255,255), 3, tipLength=0.3)

        # Info
        label = handedness[0].category_name
        lines = [
            f"Label MediaPipe: {label}",
            f"MCP: ({mcp[0]:.3f}, {mcp[1]:.3f}, {mcp[2]:.3f})",
            f"TIP: ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})",
            f"dx2d={dx2d:+.3f}  dy2d={dy2d:+.3f}",
            f"dz3d={dz3d:+.4f}  (neg=lejos camara)",
            f"mag2d={mag2d:.3f}  mag3d={mag3d:.3f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(display, line, (10, 25+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        # Imprimir en consola
        print(f"\rdx={dx2d:+.3f} dy={dy2d:+.3f} dz={dz3d:+.4f} | "
              f"apunta: {'DER' if dx2d>0 else 'IZQ'} / "
              f"{'ARR' if dy2d<0 else 'ABA'} / "
              f"{'LEJOS' if dz3d<0 else 'CERCA'}   ", end="", flush=True)

    cv2.imshow("Debug Bullet Direction", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\nListo.")
