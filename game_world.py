"""
game_world.py
Orquestador principal: escena 3D, enemigos, proyectiles, HUD, colisiones.

Mapeo de coordenadas (confirmado por debug_coords.py):
  landmarks en imagen espejada:
    lm[i].x  → 0=izq imagen=der física  … pero dx>0 = der física (confirmado)
    lm[i].y  → 0=arriba imagen, dy<0 = arriba físico

  Para renderizar con espejo natural (der física → der pantalla):
    lx = (lm[0] - 0.5) * SCALE_X    ← SIN negar (dx>0=der física=+rgt)
    ly = -(lm[1] - 0.5) * SCALE_Y   ← negar (dy<0=arriba=+Y mundo)

  Para la dirección del disparo (MCP→TIP):
    dx = tip[0]-mcp[0]  > 0 = der física = +rgt   → sin negar
    dy = tip[1]-mcp[1]  < 0 = arriba físico = +Y  → negar
"""

import math
import random
import time
import pygame

from OpenGL.GL import (
    glEnable, glDisable, glClear, glClearColor,
    glMatrixMode, glLoadIdentity,
    glLightfv, glColorMaterial, glLightModelfv,
    glBlendFunc, glDepthFunc, glShadeModel,
    glBegin, glEnd, glVertex3f, glVertex2f, glTexCoord2f, glColor3f, glColor4f,
    glLineWidth, glPointSize,
    GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_PROJECTION, GL_MODELVIEW,
    GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR,
    GL_COLOR_MATERIAL, GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LEQUAL, GL_SMOOTH, GL_LINES, GL_POINTS, GL_QUADS,
    GL_LIGHT_MODEL_AMBIENT, GL_TEXTURE_2D,
    glOrtho,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glDeleteTextures,
    GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    glPushMatrix, glPopMatrix, glTranslatef,
    glMaterialfv, GL_FRONT, GL_SHININESS,
)
from OpenGL.GLU import gluPerspective, gluNewQuadric, gluSphere, gluCylinder

from camera import Camera
from hand_tracker import HandTracker, INDEX_MCP, INDEX_TIP
from enemy import Enemy
from projectile import Projectile

# Conexiones del esqueleto de mano (MediaPipe)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Escalas compartidas entre render y disparo (deben ser idénticas)
_HAND_DEPTH   = 1.0
_HAND_SCALE_X = 1.2
_HAND_SCALE_Y = 0.9
_HAND_OFFSET_R =  0.25   # mano derecha → derecha pantalla
_HAND_OFFSET_L = -0.25   # mano izquierda → izquierda pantalla


def _lm_to_world(lm, cam, offset):
    """
    Convierte un landmark normalizado a coordenada mundo 3D.
    dx>0=derecha física → lx positivo → +rgt → SIN negar X.
    """
    lx = (lm[0] - 0.5) * _HAND_SCALE_X + offset
    ly = -(lm[1] - 0.5) * _HAND_SCALE_Y - 0.10
    fwd, rgt = cam.forward, cam.right
    return (
        cam.pos[0] + fwd[0]*_HAND_DEPTH + rgt[0]*lx,
        cam.pos[1] + ly,
        cam.pos[2] + fwd[2]*_HAND_DEPTH + rgt[2]*lx,
    )


class GameWorld:
    SPAWN_INTERVAL = 2.5
    MAX_ENEMIES    = 14
    SHOOT_COOLDOWN = 0.28

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height

        self.camera  = Camera()
        self.tracker = HandTracker()

        self._setup_gl()

        pygame.font.init()
        self._font_big   = pygame.font.SysFont("monospace", 36, bold=True)
        self._font_small = pygame.font.SysFont("monospace", 22)
        self._env_quad   = gluNewQuadric()

        self._reset_state()

    def _reset_state(self):
        self.enemies     = []
        self.projectiles = []
        self.score       = 0
        self.health      = 100
        self._spawn_timer = 0.0
        self._shoot_timer = 0.0
        self._start_time  = time.time()

    # ── Ciclo de vida ─────────────────────────────────────────────────────
    def start(self):  self.tracker.start()
    def stop(self):   self.tracker.stop()

    def restart(self):
        self._reset_state()

    # ── Update ────────────────────────────────────────────────────────────
    def update(self, dt: float):
        keys = pygame.key.get_pressed()

        if self.health <= 0:
            if keys[pygame.K_r]:
                self.restart()
            return

        # Disparo — el tracker ya emite un pulso único por flick
        self._shoot_timer -= dt
        if self.tracker.shoot and self._shoot_timer <= 0:
            self._fire_projectile()
            self._shoot_timer = self.SHOOT_COOLDOWN

        # Proyectiles
        for p in self.projectiles:
            p.update(dt)
        self.projectiles = [p for p in self.projectiles if p.alive]

        # Spawn enemigos
        self._spawn_timer += dt
        if self._spawn_timer >= self.SPAWN_INTERVAL and len(self.enemies) < self.MAX_ENEMIES:
            self.enemies.append(Enemy(self.camera.pos, self.camera.forward))
            self._spawn_timer = 0.0

        # Actualizar enemigos
        for e in self.enemies:
            e.update(dt, self.camera.pos)
            if not e.alive:
                self.health = max(0, self.health - 20)
        self.enemies = [e for e in self.enemies if e.alive]

        # Colisiones
        for p in self.projectiles[:]:
            for e in self.enemies[:]:
                if e.check_collision(p.pos):
                    p.alive = False
                    e.alive = False
                    self.score += 100
        self.projectiles = [p for p in self.projectiles if p.alive]
        self.enemies      = [e for e in self.enemies      if e.alive]

    # ── Render ────────────────────────────────────────────────────────────
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._render_3d()
        self._render_hud()

    def _render_3d(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, self.width / self.height, 0.05, 200.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.camera.apply()

        glEnable(GL_LIGHTING)
        self._setup_light()
        self._render_environment()

        for e in self.enemies:    e.render()
        for p in self.projectiles: p.render()

        glDisable(GL_LIGHTING)
        self._render_hands()
        glEnable(GL_LIGHTING)

    # ── Entorno ───────────────────────────────────────────────────────────
    def _render_environment(self):
        self._draw_floor()
        self._draw_pillars()

    def _draw_floor(self):
        glDisable(GL_LIGHTING)
        HALF, STEP, Y = 40, 2, 0.0

        glColor3f(0.82, 0.92, 0.86)
        glBegin(GL_QUADS)
        glVertex3f(-HALF, Y, -HALF); glVertex3f(HALF, Y, -HALF)
        glVertex3f( HALF, Y,  HALF); glVertex3f(-HALF, Y,  HALF)
        glEnd()

        glColor3f(0.65, 0.80, 0.72)
        glLineWidth(0.8)
        glBegin(GL_LINES)
        for i in range(-HALF, HALF+1, STEP):
            glVertex3f(i, Y+0.01, -HALF); glVertex3f(i, Y+0.01,  HALF)
            glVertex3f(-HALF, Y+0.01, i); glVertex3f( HALF, Y+0.01, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_pillars(self):
        positions = [
            (10,0,10),(-10,0,10),(10,0,-10),(-10,0,-10),
            (20,0,0),(-20,0,0),(0,0,20),(0,0,-20),
        ]
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.85,0.80,0.92,1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR,            (0.9, 0.9, 0.95,1.0))
        glMaterialfv(GL_FRONT, GL_SHININESS,           [40.0])
        for (px,_,pz) in positions:
            glPushMatrix()
            glTranslatef(px, 0.0, pz)
            gluCylinder(self._env_quad, 0.25, 0.25, 4.5, 16, 1)
            glPopMatrix()

    # ── Manos ─────────────────────────────────────────────────────────────
    def _render_hands(self):
        lm_l = self.tracker.landmarks_left
        lm_r = self.tracker.landmarks_right
        if lm_l: self._draw_hand_skeleton(lm_l, (0.5, 0.7, 1.0), _HAND_OFFSET_L)
        if lm_r: self._draw_hand_skeleton(lm_r, (0.4, 1.0, 0.6), _HAND_OFFSET_R)

    def _draw_hand_skeleton(self, landmarks, color, offset):
        """
        Renderiza esqueleto de mano.
        Mapeo: lx = (lm[0]-0.5)*SCALE_X + offset  (SIN negar X)
        Confirmado: dx>0=derecha física → lx>0 → +rgt → derecha pantalla ✓
        """
        cam = self.camera
        pts = [_lm_to_world(lm, cam, offset) for lm in landmarks]

        glColor3f(*color)
        glLineWidth(2.5)
        glBegin(GL_LINES)
        for (a, b) in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                glVertex3f(*pts[a])
                glVertex3f(*pts[b])
        glEnd()

        glPointSize(6.0)
        glBegin(GL_POINTS)
        for pt in pts: glVertex3f(*pt)
        glEnd()

    # ── Disparo ───────────────────────────────────────────────────────────
    def _fire_projectile(self):
        """
        Dirección de la bala:
          - Lateral (X/Z): normalizar dx del dedo × AMP
          - Vertical (Y):  usar dy RAW (sin normalizar) × AMP_Y separado
                           Así cuando el dedo apunta recto, dy≈0 → Y≈0

        dx confirmado: >0 = derecha física = +rgt
        dy confirmado: <0 = arriba físico  → negar para +Y mundo
        """
        cam  = self.camera
        fwd  = cam.forward
        rgt  = cam.right
        lm_r = self.tracker.landmarks_right

        if lm_r and len(lm_r) > INDEX_TIP:
            tip = lm_r[INDEX_TIP]
            mcp = lm_r[INDEX_MCP]

            dx = tip[0] - mcp[0]   # lateral: >0 = derecha
            dy = tip[1] - mcp[1]   # vertical: <0 = arriba

            # Normalizar solo X para consistencia de distancia
            # pero NO normalizar Y — queremos que sea 0 cuando el dedo apunta recto
            mag_x = abs(dx) if abs(dx) > 0.001 else 0.001
            dx_n  = dx / (abs(dx) + abs(dy) + 0.001)  # suavizar sin perder signo

            AMP_X = 1.0   # deflexión lateral
            AMP_Y = 3.0   # vertical — dy es pequeño (~0.03-0.08), necesita más amp

            raw = [
                fwd[0] + rgt[0] * dx_n * AMP_X,
                (-dy) * AMP_Y,                    # Y puro: 0 cuando dedo recto
                fwd[2] + rgt[2] * dx_n * AMP_X,
            ]
            mag = math.sqrt(sum(c*c for c in raw))
            direction = [c / mag for c in raw]
            origin = list(_lm_to_world(tip, cam, _HAND_OFFSET_R))
        else:
            origin = [cam.pos[0] + fwd[0]*0.8,
                      cam.pos[1] - 0.10,
                      cam.pos[2] + fwd[2]*0.8]
            direction = list(fwd)

        self.projectiles.append(Projectile(origin, direction))

    # ── HUD ───────────────────────────────────────────────────────────────
    def _render_hud(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self._draw_text_hud()
        if self.health <= 0:
            self._draw_game_over()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_crosshair(self):
        cx, cy = self.width//2, self.height//2
        S = 14
        glColor4f(1.0, 1.0, 1.0, 0.9)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(cx-S, cy, 0); glVertex3f(cx+S, cy, 0)
        glVertex3f(cx, cy-S, 0); glVertex3f(cx, cy+S, 0)
        glEnd()

    def _draw_text_hud(self):
        elapsed = int(time.time() - self._start_time)
        texts = [
            (f"SCORE: {self.score}",          (20, 20), (100, 240, 180)),
            (f"HEALTH: {max(0,self.health)}%", (20, 62), (255, 100, 100)),
            (f"TIME: {elapsed}s",              (20,104), (180, 180, 255)),
        ]
        if self.tracker.shoot:
            texts.append(("[ DISPARO ]", (20, 146), (255, 240, 80)))

        for text, pos, color in texts:
            surf = self._font_big.render(text, True, color)
            self._blit_surface(surf, pos)

    def _blit_surface(self, surf, pos):
        try:    data = pygame.image.tobytes(surf, "RGBA", False)
        except: data = pygame.image.tostring(surf, "RGBA", False)
        w, h = surf.get_size()

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        x, y = pos
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x,   y)
        glTexCoord2f(1, 0); glVertex2f(x+w, y)
        glTexCoord2f(1, 1); glVertex2f(x+w, y+h)
        glTexCoord2f(0, 1); glVertex2f(x,   y+h)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex])

    def _draw_game_over(self):
        glColor4f(0, 0, 0, 0.72)
        glBegin(GL_QUADS)
        glVertex2f(0, 0); glVertex2f(self.width, 0)
        glVertex2f(self.width, self.height); glVertex2f(0, self.height)
        glEnd()

        cx, cy = self.width//2, self.height//2
        s1 = self._font_big.render("GAME OVER",                              True, (255, 80,  80))
        s2 = self._font_big.render(f"Score final: {self.score}",             True, (255, 220, 80))
        s3 = self._font_small.render("R = Reiniciar   |   ESC = Salir",      True, (200, 200, 200))
        self._blit_surface(s1, (cx - s1.get_width()//2, cy - 70))
        self._blit_surface(s2, (cx - s2.get_width()//2, cy - 10))
        self._blit_surface(s3, (cx - s3.get_width()//2, cy + 50))

    # ── OpenGL setup ──────────────────────────────────────────────────────
    def _setup_gl(self):
        glClearColor(0.72, 0.85, 0.95, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.35, 0.35, 0.40, 1.0))

    def _setup_light(self):
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 10.0, 5.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (1.0, 0.95, 0.88, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0,  1.0,  1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.2, 0.2,  0.25, 1.0))
