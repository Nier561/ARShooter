# AR Survival Shooter 🎮

Prototipo de shooter de supervivencia en primera persona con **Realidad Aumentada por gestos**,
desarrollado con `pygame`, `PyOpenGL`, `opencv-python` y `mediapipe`.

## Instalación

```bash
pip install -r requirements.txt
```

> **Nota:** En Linux puede ser necesario instalar dependencias del sistema:
> ```bash
> sudo apt-get install libgl1-mesa-dev python3-opengl
> ```

## Ejecución

```bash
python main.py
```

## Controles (gestos con la webcam)

| Gesto | Acción |
|-------|--------|
| **Mano izquierda abierta, inclinada a la izquierda** | Rotar cámara izquierda |
| **Mano izquierda abierta, inclinada a la derecha** | Rotar cámara derecha |
| **Mano derecha en "pistola"** (índice extendido + pulgar arriba) | Disparar |
| **ESC** | Salir |

## Arquitectura

```
main.py          ← Bucle principal Pygame + OpenGL
camera.py        ← Cámara primera persona (Yaw)
hand_tracker.py  ← MediaPipe + OpenCV en Thread separado
enemy.py         ← Esfera roja que persigue al jugador
projectile.py    ← Proyectil 3D
game_world.py    ← Orquestador: actualización, renderizado, colisiones, HUD
requirements.txt
```

## Mecánicas

- **Enemigos** (esferas rojas) aparecen cada 3 s, máx. 12 simultáneos.
- **Disparo**: cooldown de 0.25 s para evitar ráfagas accidentales.
- **Daño**: si un enemigo llega al jugador, pierde 20 HP.
- **Puntuación**: +100 por cada enemigo eliminado.

## Rendimiento

El procesamiento de visión artificial corre en un **Thread daemon separado** (`hand_tracker.py`),
protegido con `threading.Lock`, para que el bucle de renderizado PyOpenGL mantenga
60 FPS independientemente de la carga del modelo MediaPipe.
