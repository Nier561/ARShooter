"""
camera.py
Cámara de primera persona. Controla el Yaw (rotación horizontal).
La rotación se aplica vía gluLookAt en cada frame.
"""

import math
from OpenGL.GLU import gluLookAt


class Camera:
    """
    Representa la cabeza/ojos del jugador en el mundo 3D.
    Sólo manejamos Yaw (izquierda/derecha) por ahora, pero
    la estructura está preparada para añadir Pitch fácilmente.
    """

    YAW_SPEED   = 60.0   # grados por segundo máximo
    SENSITIVITY = 0.6    # escala del input de hand-tracking [0-1]

    def __init__(self):
        # Posición fija del jugador en el mundo
        self.pos   = [0.0, 1.7, 0.0]   # altura "ojo" estándar
        self.yaw   = 0.0               # grados, 0 = mirando +Z negativo
        self.pitch = 0.0               # sin uso activo, reservado

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def apply(self):
        """Aplica la vista OpenGL. Debe llamarse después de glMatrixMode(GL_MODELVIEW)."""
        rad = math.radians(self.yaw)
        # Dirección hacia donde mira la cámara
        look_x = self.pos[0] - math.sin(rad)
        look_y = self.pos[1]
        look_z = self.pos[2] - math.cos(rad)
        gluLookAt(
            self.pos[0], self.pos[1], self.pos[2],
            look_x,      look_y,     look_z,
            0.0,         1.0,        0.0        # up-vector
        )

    @property
    def forward(self) -> list:
        """Vector unitario de la dirección de visión."""
        rad = math.radians(self.yaw)
        return [-math.sin(rad), 0.0, -math.cos(rad)]

    @property
    def right(self) -> list:
        """Vector unitario perpendicular derecho."""
        rad = math.radians(self.yaw)
        return [math.cos(rad), 0.0, -math.sin(rad)]
