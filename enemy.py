"""
enemy.py
Enemigo: esfera roja que aparece en el mundo 3D y se acerca al jugador.
"""

import math
import random
from OpenGL.GL import (
    glPushMatrix, glPopMatrix, glTranslatef, glMaterialfv,
    GL_FRONT, GL_AMBIENT_AND_DIFFUSE, GL_SPECULAR, GL_SHININESS,
    GL_EMISSION, glColor3f, glEnable, glDisable, GL_LIGHTING,
)
from OpenGL.GLU import gluNewQuadric, gluSphere


class Enemy:
    RADIUS   = 0.40
    SPEED    = 1.4
    MIN_DIST = 8.0
    MAX_DIST = 20.0

    # Rojo puro — emisivo para que no dependa de la luz
    COLOR_DIFFUSE  = (1.0, 0.05, 0.05, 1.0)
    COLOR_SPECULAR = (1.0, 0.5,  0.5,  1.0)
    COLOR_EMISSION = (0.3, 0.0,  0.0,  1.0)   # brillo propio rojo
    SHININESS      = 50.0

    def __init__(self, player_pos: list, cam_forward: list = None):
        self.pos   = self._random_spawn(player_pos, cam_forward)
        self.alive = True
        self._quad = gluNewQuadric()

    def update(self, dt: float, player_pos: list):
        dx = player_pos[0] - self.pos[0]
        dy = player_pos[1] - self.pos[1]
        dz = player_pos[2] - self.pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 0.01:
            return
        speed = self.SPEED * dt
        self.pos[0] += (dx / dist) * speed
        self.pos[1] += (dy / dist) * speed
        self.pos[2] += (dz / dist) * speed
        if dist < 0.5 + self.RADIUS:
            self.alive = False

    def render(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, self.COLOR_DIFFUSE)
        glMaterialfv(GL_FRONT, GL_SPECULAR,            self.COLOR_SPECULAR)
        glMaterialfv(GL_FRONT, GL_EMISSION,            self.COLOR_EMISSION)
        glMaterialfv(GL_FRONT, GL_SHININESS,           [self.SHININESS])
        gluSphere(self._quad, self.RADIUS, 24, 24)
        # Reset emission so other objects aren't affected
        glMaterialfv(GL_FRONT, GL_EMISSION, (0.0, 0.0, 0.0, 1.0))
        glPopMatrix()

    def check_collision(self, proj_pos: list) -> bool:
        dx = proj_pos[0] - self.pos[0]
        dy = proj_pos[1] - self.pos[1]
        dz = proj_pos[2] - self.pos[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz) < self.RADIUS + 0.15

    @staticmethod
    def _random_spawn(player_pos: list, cam_forward: list = None) -> list:
        """
        Spawn distribuido uniformemente en un arco frontal de ±80°.
        Usa el forward real de la cámara para que siempre vengan de frente.
        """
        import math, random
        dist = random.uniform(Enemy.MIN_DIST, Enemy.MAX_DIST)

        # Ángulo base = dirección que mira la cámara en XZ
        if cam_forward is not None:
            base_angle = math.atan2(cam_forward[0], cam_forward[2])
        else:
            base_angle = math.pi   # fallback: -Z

        spread = math.radians(35)
        angle  = base_angle + random.uniform(-spread, spread)
        y_off  = random.uniform(0.8, 2.5)

        return [
            player_pos[0] + dist * math.sin(angle),
            player_pos[1] + y_off,
            player_pos[2] + dist * math.cos(angle),
        ]
