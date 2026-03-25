"""
fix_mediapipe.py
Detecta la versión exacta de mediapipe instalada y descarga el modelo correcto.
Ejecutar: python fix_mediapipe.py
"""
import subprocess, sys, os, tempfile, urllib.request, zipfile

# ─── 1. Versión instalada ──────────────────────────────────────────────────
import mediapipe as mp
version = mp.__version__
print(f"[INFO] mediapipe instalado: {version}")
major, minor, patch = (int(x) for x in version.split(".")[:3])
print(f"[INFO] Python: {sys.version}")

# ─── 2. Borrar modelo existente siempre ────────────────────────────────────
model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
if os.path.exists(model_path):
    os.remove(model_path)
    print(f"[INFO] Borrado modelo anterior: {model_path}")

# ─── 3. Elegir URL según versión ──────────────────────────────────────────
# mediapipe 0.10.x cambió el formato del modelo varias veces.
# Las versiones más nuevas (0.10.14+) requieren el modelo "bundle" diferente.
MODELS = {
    "new":    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "legacy": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

urls_to_try = [MODELS["new"], MODELS["legacy"]]

def try_model(url, path):
    """Descarga y valida que MediaPipe pueda abrir el modelo."""
    print(f"\n[TEST] Descargando: {url}")
    try:
        def prog(c, b, t):
            pct = min(c*b*100//t, 100)
            print(f"\r  {pct}%", end="", flush=True)
        urllib.request.urlretrieve(url, path, prog)
        print(f"\n  Tamaño: {os.path.getsize(path)//1024} KB")
    except Exception as e:
        print(f"  ✗ Descarga fallida: {e}")
        return False

    # Intentar cargar con MediaPipe
    try:
        from mediapipe.tasks import python as _mpt
        from mediapipe.tasks.python import vision as _mpv
        base = _mpt.BaseOptions(model_asset_path=path)
        opts = _mpv.HandLandmarkerOptions(
            base_options=base,
            num_hands=1,
            running_mode=_mpv.RunningMode.IMAGE,
        )
        det = _mpv.HandLandmarker.create_from_options(opts)
        det.close()
        print(f"  ✓ Modelo compatible con mediapipe {version}")
        return True
    except Exception as e:
        print(f"  ✗ No compatible: {e}")
        if os.path.exists(path):
            os.remove(path)
        return False

ok = False
for url in urls_to_try:
    if try_model(url, model_path):
        ok = True
        break

# ─── 4. Si ningún modelo funciona, intentar downgrade ─────────────────────
if not ok:
    print("\n" + "="*60)
    print("SOLUCIÓN: Ningún modelo es compatible con tu versión de mediapipe.")
    print(f"  Versión actual: mediapipe {version}")
    print()
    print("Opciones (elige una):")
    print()
    print("  OPCIÓN A — Bajar mediapipe a versión estable probada:")
    print("    pip install mediapipe==0.10.9")
    print()
    print("  OPCIÓN B — Subir a la última versión:")
    print("    pip install --upgrade mediapipe")
    print()
    print("  Luego vuelve a ejecutar: python fix_mediapipe.py")
    print("="*60)
    sys.exit(1)

# ─── 5. Test de cámara ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Modelo OK. Buscando cámara...")

import cv2

BACKENDS = [
    ("CAP_MSMF",    cv2.CAP_MSMF),     # Microsoft Media Foundation — mejor en Win11
    ("CAP_ANY",     cv2.CAP_ANY),       # OpenCV elige automáticamente
    ("CAP_DSHOW",   cv2.CAP_DSHOW),     # DirectShow
    ("CAP_VFW",     cv2.CAP_VFW),       # Video for Windows (legacy)
]

found       = None
found_backend = None
for name, backend in BACKENDS:
    print(f"\n  Probando backend {name}:")
    for idx in range(4):
        try:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"    ✓ idx={idx}  {w}x{h}  ← FUNCIONA")
                    if found is None:
                        found         = idx
                        found_backend = name
                else:
                    print(f"    ⚠ idx={idx}  abre pero sin frames")
            else:
                cap.release()
        except Exception as e:
            print(f"    ✗ idx={idx}  {e}")

if found is None:
    print("\n✗ No hay cámara disponible con ningún backend.")
    print()
    print("  Posibles causas en Windows:")
    print("  1. La cámara está siendo usada por el sistema (Windows Hello/face login)")
    print("     → Ve a Configuración > Privacidad > Cámara y asegúrate de que esté habilitada para apps de escritorio")
    print("  2. Driver no instalado correctamente")
    print("     → Abre Administrador de dispositivos → Dispositivos de imagen → verifica que aparezca tu cámara")
    print("  3. La cámara es virtual (OBS, ManyCam) y necesita otro índice")
    print()
    print("  Prueba manual — ejecuta esto en Python:")
    print("    import cv2")
    print("    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)")
    print("    print(cap.isOpened(), cap.read()[0])")
else:
    cfg = os.path.join(os.path.dirname(__file__), "camera_config.txt")
    # Guardar índice y backend
    with open(cfg, "w") as f:
        f.write(f"{found}\n{found_backend}")
    print(f"\n✓ Cámara {found} con backend {found_backend} guardada en camera_config.txt")

print("\n✓ Todo listo. Ejecuta: python main.py")
