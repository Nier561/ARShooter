"""
hand_tracker.py
Captura de cámara web + MediaPipe en hilo dedicado.

Sistema de coordenadas confirmado por debug_coords.py:
  Después de cv2.flip(frame,1):
    lm[i].x  → 0 = izquierda imagen = DERECHA física (flip espeja)
                ESPERA: tras el flip x=0 es izq imagen = der física?
                NO. flip horizontal: lo que estaba a la izquierda va a la derecha.
                Resultado: x=0 imagen espejada = derecha de la cámara = DERECHA física.
                Confirmado: dx = tip.x - mcp.x > 0 cuando apuntas a la DERECHA física ✓

    lm[i].y  → 0 = arriba imagen. dy < 0 cuando apuntas ARRIBA físico ✓

  Tasks API con imagen espejada invierte handedness:
    Reporta "Left"  → mano DERECHA física  → corregimos a "Right"
    Reporta "Right" → mano IZQUIERDA física → corregimos a "Left"

  Legacy API corrige handedness automáticamente → no tocar.
"""

import threading
import time as _time
import cv2
import numpy as np

# ── Índices de landmarks MediaPipe Hands ─────────────────────────────────
WRIST      = 0
THUMB_CMC  = 1
THUMB_TIP  = 4
INDEX_MCP  = 5
INDEX_PIP  = 6
INDEX_TIP  = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP   = 13
RING_PIP   = 14
RING_TIP   = 16
PINKY_MCP  = 17
PINKY_PIP  = 18
PINKY_TIP  = 20


class HandTracker:
    """
    Corre MediaPipe Hands en un Thread separado.
    El hilo principal solo lee propiedades protegidas por Lock.
    """

    def __init__(self, camera_index: int = 0):
        import platform, os

        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_config.txt")
        backend = cv2.CAP_MSMF if platform.system() == "Windows" else cv2.CAP_ANY

        if os.path.exists(config_path):
            try:
                lines = open(config_path).read().strip().splitlines()
                camera_index = int(lines[0])
                if len(lines) > 1:
                    backend_map = {
                        "CAP_MSMF":  cv2.CAP_MSMF,
                        "CAP_ANY":   cv2.CAP_ANY,
                        "CAP_DSHOW": cv2.CAP_DSHOW,
                        "CAP_VFW":   cv2.CAP_VFW,
                    }
                    backend = backend_map.get(lines[1].strip(), backend)
                print(f"[HandTracker] Config: cam={camera_index}  backend={lines[1] if len(lines)>1 else 'default'}")
            except Exception as e:
                print(f"[HandTracker] Error leyendo config: {e}")

        self._cap = cv2.VideoCapture(camera_index, backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        if not self._cap.isOpened():
            print(f"[HandTracker] ✗ No se pudo abrir cámara {camera_index}")
        else:
            print(f"[HandTracker] ✓ Cámara {camera_index} abierta")

        self._lock = threading.Lock()
        self._stop = threading.Event()

        # Estado compartido
        self._shoot           = False
        self._landmarks_left  = []   # lista de (x,y,z) normalizados en imagen espejada
        self._landmarks_right = []

        self._thread = threading.Thread(target=self._run, daemon=True)

    # ── Ciclo de vida ─────────────────────────────────────────────────────
    def start(self):  self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()

    # ── Propiedades thread-safe ───────────────────────────────────────────
    @property
    def shoot(self) -> bool:
        with self._lock: return self._shoot

    @property
    def landmarks_left(self) -> list:
        with self._lock: return list(self._landmarks_left)

    @property
    def landmarks_right(self) -> list:
        with self._lock: return list(self._landmarks_right)

    # ── Hilo de captura ───────────────────────────────────────────────────
    def _run(self):
        if not self._cap.isOpened():
            print("[HandTracker] ✗ Sin cámara. Ejecuta fix_mediapipe.py")
            return

        hands, use_legacy = self._init_mediapipe()
        if hands is None:
            return

    def _run(self):
        if not self._cap.isOpened():
            print("[HandTracker] ✗ Sin cámara. Ejecuta fix_mediapipe.py")
            return

        hands, use_legacy = self._init_mediapipe()
        if hands is None:
            return

        # Para detección de flick: historial de posición Y del TIP del índice
        _tip_y_history = []   # lista de (timestamp, y)
        _HISTORY_WINDOW = 0.12   # segundos hacia atrás que miramos
        _FLICK_THRESH   = 0.18   # velocidad mínima en unidades/seg para disparar
        _FLICK_COOLDOWN = 0.35   # segundos mínimos entre disparos
        _last_flick_time = 0.0
        _last_ts = 0  # último timestamp enviado a MediaPipe

        frame_count = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                _time.sleep(0.01)
                continue

            flipped = cv2.flip(frame, 1)
            rgb     = np.ascontiguousarray(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))

            shoot, lm_left, lm_right = False, [], []

            try:
                if use_legacy:
                    results = hands.process(rgb)
                    for hand_lm, handedness in zip(
                        results.multi_hand_landmarks or [],
                        results.multi_handedness     or [],
                    ):
                        label = handedness.classification[0].label
                        lm    = [(p.x, p.y, p.z) for p in hand_lm.landmark]
                        lm_left, lm_right = self._assign_hand(
                            label, lm, lm_left, lm_right, use_legacy=True)
                else:
                    import mediapipe as _mp
                    # Timestamp estrictamente creciente (microsegundos → ms)
                    ts = int(_time.monotonic() * 1000)
                    if ts <= _last_ts:
                        ts = _last_ts + 1
                    _last_ts = ts
                    img   = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
                    res   = hands.detect_for_video(img, ts)
                    for hand_lm, handedness in zip(res.hand_landmarks, res.handedness):
                        label = handedness[0].category_name
                        lm    = [(p.x, p.y, p.z) for p in hand_lm]
                        lm_left, lm_right = self._assign_hand(
                            label, lm, lm_left, lm_right, use_legacy=False)

                # Detección de flick con la mano derecha
                now = _time.monotonic()
                if lm_right and len(lm_right) > INDEX_TIP:
                    tip_y = lm_right[INDEX_TIP][1]

                    # Acumular historial
                    _tip_y_history.append((now, tip_y))
                    # Limpiar entradas viejas
                    _tip_y_history = [(t, y) for t, y in _tip_y_history
                                      if now - t <= _HISTORY_WINDOW]

                    # Calcular velocidad: dy/dt en el ventana
                    if len(_tip_y_history) >= 2:
                        t0, y0 = _tip_y_history[0]
                        t1, y1 = _tip_y_history[-1]
                        dt_hist = t1 - t0
                        if dt_hist > 0.01:
                            # dy negativo = TIP sube (coordenadas imagen: 0=arriba)
                            velocity = (y1 - y0) / dt_hist

                            # Índice extendido = TIP más arriba que PIP
                            index_extended = (lm_right[INDEX_TIP][1] <
                                              lm_right[INDEX_PIP][1] - 0.02)

                            # Flick: velocidad negativa fuerte + índice extendido + cooldown
                            if (velocity < -_FLICK_THRESH and
                                index_extended and
                                now - _last_flick_time > _FLICK_COOLDOWN):
                                shoot = True
                                _last_flick_time = now
                                _tip_y_history.clear()   # reset para evitar doble disparo
                else:
                    _tip_y_history.clear()

                frame_count += 1
                if frame_count % 120 == 0:
                    n = (1 if lm_left else 0) + (1 if lm_right else 0)
                    print(f"[HandTracker] f={frame_count}  manos={n}  shoot={shoot}")

            except Exception as ex:
                print(f"[HandTracker] Error frame: {ex}")

            with self._lock:
                self._shoot           = shoot
                self._landmarks_left  = lm_left
                self._landmarks_right = lm_right

        hands.close()

    def _init_mediapipe(self):
        """Intenta legacy API primero, luego Tasks API. Retorna (hands, use_legacy)."""
        # Legacy
        try:
            import mediapipe as _mp
            _h = getattr(getattr(_mp, "solutions", None), "hands", None)
            if _h is None: raise AttributeError()
            hands = _h.Hands(static_image_mode=False, max_num_hands=2,
                             min_detection_confidence=0.6, min_tracking_confidence=0.5)
            print("[HandTracker] ✓ Legacy API (solutions.hands)")
            return hands, True
        except Exception as e:
            print(f"[HandTracker] Legacy no disponible: {e}")

        # Tasks API
        try:
            from mediapipe.tasks import python as _mpt
            from mediapipe.tasks.python import vision as _mpv
            import os, tempfile, zipfile, urllib.request

            MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                          "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")

            model_ok = False
            if os.path.exists(model_path) and os.path.getsize(model_path) > 500_000:
                try:
                    zipfile.ZipFile(model_path).testzip()
                    model_ok = True
                    print(f"[HandTracker] Modelo OK ({os.path.getsize(model_path)//1024} KB)")
                except Exception:
                    os.remove(model_path)

            if not model_ok:
                print("[HandTracker] Descargando modelo…")
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print(f"[HandTracker] ✓ Descargado ({os.path.getsize(model_path)//1024} KB)")

            opts = _mpv.HandLandmarkerOptions(
                base_options=_mpt.BaseOptions(model_asset_path=model_path),
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=_mpv.RunningMode.VIDEO,
            )
            hands = _mpv.HandLandmarker.create_from_options(opts)
            print("[HandTracker] ✓ Tasks API VIDEO mode")
            return hands, False
        except Exception as e:
            print(f"[HandTracker] ERROR fatal: {e}")
            import traceback; traceback.print_exc()
            return None, False

    @staticmethod
    def _assign_hand(label, lm, lm_left, lm_right, use_legacy):
        """
        Asigna landmarks a la mano correcta.
        Tasks API con imagen espejada invierte los labels → corregimos.
        """
        if not use_legacy:
            label = "Right" if label == "Left" else "Left"

        if label == "Right":
            lm_right = lm
        else:
            lm_left = lm

        return lm_left, lm_right
