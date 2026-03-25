"""
projectile.py
Proyectil disparado desde la mano derecha hacia el centro de la pantalla.
"""

import math
from OpenGL.GL import (
    glPushMatrix, glPopMatrix, glTranslatef,
    glMaterialfv, GL_FRONT, GL_AMBIENT_AND_DIFFUSE,
    GL_SPECULAR, GL_SHININESS,
)
from OpenGL.GLU import gluNewQuadric, gluSphere


class Projectile:
    """
    Esfera pequeña que viaja en línea recta hasta su distancia máxima.
    """

    SPEED    = 18.0   # unidades/segundo
    MAX_DIST = 40.0   # distancia máxima antes de desaparecer
    RADIUS   = 0.12

    COLOR_DIFFUSE  = (0.4, 0.85, 1.0, 1.0)   # azul celeste suave
    COLOR_SPECULAR = (1.0, 1.0,  1.0, 1.0)
    SHININESS      = 90.0

    def __init__(self, origin: list, direction: list):
        self.pos      = list(origin)
        self.dir      = self._normalize(direction)
        self.traveled = 0.0
        self.alive    = True
        self._quad    = gluNewQuadric()

    # ------------------------------------------------------------------
    def update(self, dt: float):
        step = self.SPEED * dt
        self.pos[0] += self.dir[0] * step
        self.pos[1] += self.dir[1] * step
        self.pos[2] += self.dir[2] * step
        self.traveled += step
        if self.traveled >= self.MAX_DIST:
            self.alive = False

    def render(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, self.COLOR_DIFFUSE)
        glMaterialfv(GL_FRONT, GL_SPECULAR,            self.COLOR_SPECULAR)
        glMaterialfv(GL_FRONT, GL_SHININESS,           [self.SHININESS])
        gluSphere(self._quad, self.RADIUS, 12, 12)
        glPopMatrix()

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(v: list) -> list:
        mag = math.sqrt(sum(c*c for c in v))
        if mag < 1e-6:
            return [0.0, 0.0, -1.0]
        return [c / mag for c in v]
