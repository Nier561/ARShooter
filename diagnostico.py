"""
diagnostico.py
Ejecuta este script ANTES del juego para verificar:
1. Que la cámara abre correctamente
2. Que el modelo de MediaPipe descarga e inicializa bien
3. Que la detección de manos funciona

Uso: python diagnostico.py
"""

import os
import sys
import tempfile


# ─── 1. LIMPIAR MODELO CORRUPTO ────────────────────────────────────────────
model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
if os.path.exists(model_path):
    size_kb = os.path.getsize(model_path) // 1024
    print(f"[1] Modelo encontrado: {model_path}  ({size_kb} KB)")
    if size_kb < 500:
        print(f"    ⚠  Archivo corrupto/incompleto ({size_kb} KB < 500 KB). Eliminando…")
        os.remove(model_path)
        print("    ✓  Eliminado.")
    else:
        print(f"    ✓  Tamaño OK ({size_kb} KB)")
else:
    print(f"[1] Modelo no existe aún: {model_path}")


# ─── 2. DESCARGAR MODELO ───────────────────────────────────────────────────
if not os.path.exists(model_path):
    import urllib.request

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    print(f"\n[2] Descargando modelo desde:\n    {MODEL_URL}")
    print("    (esto puede tardar 10-30 seg según tu conexión…)")

    try:
        def progress(count, block_size, total_size):
            pct = min(count * block_size * 100 // total_size, 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r    [{bar}] {pct}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=progress)
        print()
        size_kb = os.path.getsize(model_path) // 1024
        print(f"    ✓  Descarga completa: {size_kb} KB")
    except Exception as e:
        print(f"\n    ✗  ERROR descargando: {e}")
        print("    → Verifica tu conexión a internet e intenta de nuevo.")
        sys.exit(1)
else:
    print("[2] Modelo ya existe, skip descarga.")


# ─── 3. INICIALIZAR MEDIAPIPE ──────────────────────────────────────────────
print("\n[3] Inicializando MediaPipe HandLandmarker…")
try:
    from mediapipe.tasks import python as _mpt
    from mediapipe.tasks.python import vision as _mpv

    base_opts = _mpt.BaseOptions(model_asset_path=model_path)
    opts = _mpv.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=_mpv.RunningMode.IMAGE,
    )
    detector = _mpv.HandLandmarker.create_from_options(opts)
    print("    ✓  MediaPipe OK")
except Exception as e:
    print(f"    ✗  ERROR MediaPipe: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ─── 4. BUSCAR CÁMARA DISPONIBLE ──────────────────────────────────────────
print("\n[4] Buscando cámaras disponibles…")
import cv2

found_index = None
for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # CAP_DSHOW es más fiable en Windows
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"    ✓  Cámara {idx}: {w}x{h}  ← USABLE")
            if found_index is None:
                found_index = idx
        else:
            print(f"    ⚠  Cámara {idx}: abre pero no devuelve frames")
        cap.release()
    else:
        print(f"    ✗  Cámara {idx}: no disponible")

if found_index is None:
    print("\n    ✗  No se encontró ninguna cámara funcional.")
    print("    Posibles causas:")
    print("      - Otra app (Teams, Zoom, navegador) está usando la cámara → ciérrala")
    print("      - Driver de cámara no instalado")
    print("      - Cámara deshabilitada en el Administrador de dispositivos")
    sys.exit(1)

print(f"\n    → Usando cámara índice: {found_index}")


# ─── 5. TEST DE DETECCIÓN EN VIVO ─────────────────────────────────────────
print("\n[5] Abriendo ventana de prueba (muestra tu mano, presiona Q para salir)…")

import numpy as np
import mediapipe as _mp

cap = cv2.VideoCapture(found_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_drawing       = None
mp_drawing_styles = None
# Intentar usar utils de dibujo si están disponibles
try:
    mp_drawing        = _mp.solutions.drawing_utils
    mp_drawing_styles = _mp.solutions.drawing_styles
    mp_hands_mod      = _mp.solutions.hands
except:
    pass

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

frame_count = 0
hands_detected_total = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("  ⚠  No se pudo leer frame")
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_c   = np.ascontiguousarray(rgb)
    mp_img  = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb_c)
    result  = detector.detect(mp_img)

    n_hands = len(result.hand_landmarks)
    if n_hands > 0:
        hands_detected_total += 1

    # Dibujar landmarks
    for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name
        pts   = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0]))
                 for p in hand_lm]
        color = (0, 200, 100) if label == "Right" else (100, 100, 255)
        for (a, b) in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.putText(frame, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    # HUD
    status = f"Manos: {n_hands}  |  Total detectadas: {hands_detected_total}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (50, 220, 50), 2)
    cv2.putText(frame, "Q = salir", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Diagnostico AR Shooter", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()

print(f"\n[5] Resumen: {frame_count} frames procesados, manos detectadas en {hands_detected_total} frames")

# ─── 6. GUARDAR ÍNDICE DE CÁMARA ──────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), "camera_config.txt")
with open(config_path, "w") as f:
    f.write(str(found_index))
print(f"\n[6] Índice de cámara ({found_index}) guardado en: {config_path}")
print("\n✓  Diagnóstico completo. Ahora ejecuta: python main.py")
