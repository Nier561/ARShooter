"""
projectile.py
Proyectil láser con estela luminosa.
"""

import math
from collections import deque
from OpenGL.GL import (
    glPushMatrix, glPopMatrix, glTranslatef,
    glMaterialfv, GL_FRONT, GL_AMBIENT_AND_DIFFUSE,
    GL_SPECULAR, GL_SHININESS, GL_EMISSION,
    glBegin, glEnd, glVertex3f, glColor4f, glLineWidth,
    glDisable, glEnable, GL_LIGHTING, GL_LINES,
)
from OpenGL.GLU import gluNewQuadric, gluSphere


class Projectile:
    """
    Esfera luminosa que viaja en línea recta con estela.
    """

    SPEED    = 18.0
    MAX_DIST = 40.0
    RADIUS   = 0.10

    # Cyan/verde brillante — estilo láser espacial
    COLOR_DIFFUSE  = (0.2, 1.0, 0.6, 1.0)
    COLOR_SPECULAR = (1.0, 1.0, 1.0, 1.0)
    COLOR_EMISSION = (0.1, 0.8, 0.4, 1.0)   # brillo propio verde
    SHININESS      = 100.0

    TRAIL_LENGTH   = 6   # número de posiciones en la estela

    def __init__(self, origin: list, direction: list):
        self.pos      = list(origin)
        self.dir      = self._normalize(direction)
        self.traveled = 0.0
        self.alive    = True
        self._quad    = gluNewQuadric()
        self._trail   = deque(maxlen=self.TRAIL_LENGTH)

    def update(self, dt: float):
        # Guardar posición actual en la estela
        self._trail.append(tuple(self.pos))

        step = self.SPEED * dt
        self.pos[0] += self.dir[0] * step
        self.pos[1] += self.dir[1] * step
        self.pos[2] += self.dir[2] * step
        self.traveled += step
        if self.traveled >= self.MAX_DIST:
            self.alive = False

    def render(self):
        # ── Estela ──
        if len(self._trail) >= 2:
            glDisable(GL_LIGHTING)
            glLineWidth(2.5)
            glBegin(GL_LINES)
            trail_list = list(self._trail)
            for i in range(len(trail_list) - 1):
                # Alpha se desvanece hacia el inicio de la estela
                alpha = (i + 1) / len(trail_list)
                glColor4f(0.1, 0.9 * alpha, 0.4 * alpha, alpha * 0.7)
                glVertex3f(*trail_list[i])
                glColor4f(0.1, 0.9 * alpha, 0.4 * alpha, alpha * 0.85)
                glVertex3f(*trail_list[i + 1])
            glEnd()
            glEnable(GL_LIGHTING)

        # ── Esfera principal ──
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, self.COLOR_DIFFUSE)
        glMaterialfv(GL_FRONT, GL_SPECULAR,            self.COLOR_SPECULAR)
        glMaterialfv(GL_FRONT, GL_SHININESS,           [self.SHININESS])
        glMaterialfv(GL_FRONT, GL_EMISSION,            self.COLOR_EMISSION)
        gluSphere(self._quad, self.RADIUS, 12, 12)
        # Reset emission
        glMaterialfv(GL_FRONT, GL_EMISSION, (0.0, 0.0, 0.0, 1.0))
        glPopMatrix()

    @staticmethod
    def _normalize(v: list) -> list:
        mag = math.sqrt(sum(c*c for c in v))
        if mag < 1e-6:
            return [0.0, 0.0, -1.0]
        return [c / mag for c in v]
