"""
game_world.py
Orquestador principal: escena 3D espacial, asteroides, proyectiles, HUD, colisiones.

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
from hand_tracker import HandTracker, INDEX_MCP, INDEX_TIP, WRIST
from asteroid import Asteroid
from projectile import Projectile
from sounds import SoundManager

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


# ── Partícula de explosión ────────────────────────────────────────────────
class _Particle:
    """Fragmento de explosión de asteroide."""
    def __init__(self, pos, color):
        self.pos = list(pos)
        speed = random.uniform(2.0, 6.0)
        # Dirección aleatoria esférica
        theta = random.uniform(0, 2 * math.pi)
        phi   = random.uniform(0, math.pi)
        self.vel = [
            speed * math.sin(phi) * math.cos(theta),
            speed * math.sin(phi) * math.sin(theta),
            speed * math.cos(phi),
        ]
        self.life     = random.uniform(0.3, 0.7)
        self.max_life = self.life
        self.size     = random.uniform(2.0, 5.0)
        self.color    = color
        self.alive    = True

    def update(self, dt):
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.pos[2] += self.vel[2] * dt
        # Decelerar
        self.vel[0] *= 0.96
        self.vel[1] *= 0.96
        self.vel[2] *= 0.96
        self.life -= dt
        if self.life <= 0:
            self.alive = False


# Game states
STATE_MENU    = "menu"
STATE_PLAYING = "playing"
STATE_OVER    = "game_over"


class GameWorld:
    SPAWN_INTERVAL = 2.0
    MAX_ENEMIES    = 14
    SHOOT_COOLDOWN = 0.28

    # Puntos por tamaño de asteroide
    SCORE_MAP = {"small": 50, "medium": 100, "large": 200}

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height

        self.camera  = Camera()
        self.tracker = HandTracker()
        self.sounds  = SoundManager()

        self._setup_gl()

        pygame.font.init()
        self._font_hero   = pygame.font.SysFont("consolas", 56, bold=True)
        self._font_title  = pygame.font.SysFont("consolas", 42, bold=True)
        self._font_big    = pygame.font.SysFont("consolas", 28, bold=True)
        self._font_medium = pygame.font.SysFont("consolas", 22, bold=True)
        self._font_small  = pygame.font.SysFont("consolas", 18)
        self._font_label  = pygame.font.SysFont("consolas", 14)
        self._font_tiny   = pygame.font.SysFont("consolas", 13)

        # Generar campo de estrellas (una sola vez)
        self._stars = self._generate_starfield(500)

        self.state = STATE_MENU
        self._menu_hover = None   # "start" or "exit" or None
        self._menu_star_angle = 0.0  # para animación sutil del menú
        self._reset_state()

    def _generate_starfield(self, count):
        """Genera estrellas distribuidas en una esfera grande alrededor del origen."""
        stars = []
        for _ in range(count):
            theta = random.uniform(0, 2 * math.pi)
            phi   = random.uniform(0, math.pi)
            r     = random.uniform(60, 100)
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            size = random.uniform(1.0, 3.0)
            brightness = random.uniform(0.5, 1.0)
            # Tipo de color
            ctype = random.random()
            if ctype < 0.6:
                color = (brightness, brightness, brightness)               # blanca
            elif ctype < 0.8:
                color = (brightness * 0.7, brightness * 0.8, brightness)   # azulada
            else:
                color = (brightness, brightness * 0.9, brightness * 0.6)   # amarillenta
            stars.append((x, y, z, size, color))
        return stars

    def _reset_state(self):
        self.enemies     = []   # ahora son Asteroid
        self.projectiles = []
        self.particles   = []   # partículas de explosión
        self.score       = 0
        self.health      = 100
        self._spawn_timer = 0.0
        self._shoot_timer = 0.0
        self._start_time  = time.time()

    # ── Ciclo de vida ─────────────────────────────────────────────────────
    def start(self):
        self.tracker.start()
        # No iniciar sonido ambiente hasta que comience el juego

    def stop(self):
        self.tracker.stop()
        self.sounds.stop_ambient()

    def restart(self):
        self._reset_state()
        self.state = STATE_PLAYING
        self.sounds.start_ambient()

    def start_game(self):
        """Transición de menú a juego."""
        self._reset_state()
        self.state = STATE_PLAYING
        self.sounds.start_ambient()
        pygame.mouse.set_visible(False)

    def back_to_menu(self):
        """Vuelve al menú principal."""
        self.state = STATE_MENU
        self._menu_hover = None
        self.sounds.stop_ambient()
        pygame.mouse.set_visible(True)

    def handle_event(self, event):
        """Procesa eventos de pygame (mouse clicks, etc)."""
        if self.state == STATE_MENU:
            if event.type == pygame.MOUSEMOTION:
                self._menu_hover = self._get_menu_button(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                btn = self._get_menu_button(event.pos)
                if btn == "start":
                    self.start_game()
                elif btn == "exit":
                    return "quit"
        elif self.state == STATE_OVER:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.restart()
                elif event.key == pygame.K_m:
                    self.back_to_menu()
        return None

    def _get_menu_button(self, mouse_pos):
        """Devuelve qué botón del menú está bajo el cursor."""
        mx, my = mouse_pos
        cx = self.width // 2
        # Botones centrados
        btn_w, btn_h = 220, 48
        btn_x = cx - btn_w // 2
        # Start button Y
        start_y = self.height // 2 + 10
        if btn_x <= mx <= btn_x + btn_w and start_y <= my <= start_y + btn_h:
            return "start"
        # Exit button Y
        exit_y = start_y + btn_h + 16
        if btn_x <= mx <= btn_x + btn_w and exit_y <= my <= exit_y + btn_h:
            return "exit"
        return None

    # ── Update ────────────────────────────────────────────────────────────
    def update(self, dt: float):
        if self.state == STATE_MENU:
            self._menu_star_angle += dt * 2.0  # rotación sutil
            return

        if self.state == STATE_OVER:
            return

        keys = pygame.key.get_pressed()

        # Rotacion con mano izquierda
        lm_l = self.tracker.landmarks_left
        if lm_l and len(lm_l) > WRIST:
            wrist_x = lm_l[WRIST][0]  # 0-1, centro ~0.5
            # Zona muerta en el centro (0.35 - 0.65)
            if wrist_x < 0.35:
                # Mano a la izquierda de la imagen = derecha fisica → girar derecha
                intensity = (0.35 - wrist_x) / 0.35  # 0..1
                self.camera.rotate(intensity, dt)
            elif wrist_x > 0.65:
                # Mano a la derecha de la imagen = izquierda fisica → girar izquierda
                intensity = (wrist_x - 0.65) / 0.35  # 0..1
                self.camera.rotate(-intensity, dt)

        # Disparo — el tracker ya emite un pulso único por flick
        self._shoot_timer -= dt
        if self.tracker.shoot and self._shoot_timer <= 0:
            self._fire_projectile()
            self._shoot_timer = self.SHOOT_COOLDOWN
            self.sounds.play_shoot()

        # Proyectiles
        for p in self.projectiles:
            p.update(dt)
        self.projectiles = [p for p in self.projectiles if p.alive]

        # Spawn asteroides
        self._spawn_timer += dt
        if self._spawn_timer >= self.SPAWN_INTERVAL and len(self.enemies) < self.MAX_ENEMIES:
            size = random.choices(
                ["small", "medium", "large"],
                weights=[0.4, 0.4, 0.2],
            )[0]
            self.enemies.append(
                Asteroid(self.camera.pos, self.camera.forward, size_category=size)
            )
            self._spawn_timer = 0.0

        # Actualizar asteroides
        for e in self.enemies:
            e.update(dt, self.camera.pos)
            if not e.alive:
                self.health = max(0, self.health - 20)
                self.sounds.play_hit()
                self._spawn_particles(e.pos, (1.0, 0.3, 0.1), count=6)
        self.enemies = [e for e in self.enemies if e.alive]

        # Transición a game over
        if self.health <= 0:
            self.state = STATE_OVER
            self.sounds.stop_ambient()
            pygame.mouse.set_visible(True)
            return

        # Colisiones proyectil-asteroide
        for p in self.projectiles[:]:
            for e in self.enemies[:]:
                if e.check_collision(p.pos):
                    p.alive = False
                    e.alive = False
                    pts = self.SCORE_MAP.get(e.size_category, 100)
                    self.score += pts
                    self.sounds.play_explosion()
                    # Partículas de explosión
                    self._spawn_particles(
                        e.pos,
                        (0.9, 0.6, 0.2),
                        count=10 if e.size_category != "small" else 6,
                    )
        self.projectiles = [p for p in self.projectiles if p.alive]
        self.enemies     = [e for e in self.enemies     if e.alive]

        # Actualizar partículas
        for part in self.particles:
            part.update(dt)
        self.particles = [p for p in self.particles if p.alive]

    def _spawn_particles(self, pos, base_color, count=8):
        """Genera partículas de explosión en una posición."""
        for _ in range(count):
            # Variación del color
            color = (
                min(1, base_color[0] + random.uniform(-0.15, 0.15)),
                min(1, base_color[1] + random.uniform(-0.15, 0.15)),
                min(1, base_color[2] + random.uniform(-0.1, 0.1)),
            )
            self.particles.append(_Particle(pos, color))

    # ── Render ────────────────────────────────────────────────────────────
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.state == STATE_MENU:
            self._render_menu_bg()
            self._render_menu()
        else:
            self._render_3d()
            self._render_hud()

    def _render_3d(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, self.width / self.height, 0.05, 200.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.camera.apply()

        # Estrellas (sin iluminación)
        self._render_starfield()

        glEnable(GL_LIGHTING)
        self._setup_light()

        for e in self.enemies:     e.render()
        for p in self.projectiles: p.render()

        glDisable(GL_LIGHTING)

        # Partículas de explosión (sin iluminación)
        self._render_particles()

        # Manos
        self._render_hands()

        glEnable(GL_LIGHTING)

    # ── Estrellas ─────────────────────────────────────────────────────────
    def _render_starfield(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)  # estrellas siempre al fondo

        cx, cy, cz = self.camera.pos

        # Agrupar por tamaño para evitar llamar glPointSize dentro de glBegin
        buckets = {}
        for (x, y, z, size, color) in self._stars:
            # Redondear tamaño para agrupar
            s = round(size * 2) / 2  # 0.5 incrementos
            if s not in buckets:
                buckets[s] = []
            buckets[s].append((cx + x, cy + y, cz + z, color))

        for pt_size, star_list in buckets.items():
            glPointSize(pt_size)
            glBegin(GL_POINTS)
            for (sx, sy, sz, color) in star_list:
                glColor3f(*color)
                glVertex3f(sx, sy, sz)
            glEnd()

        glEnable(GL_DEPTH_TEST)

    # ── Partículas ────────────────────────────────────────────────────────
    def _render_particles(self):
        if not self.particles:
            return
        glDisable(GL_LIGHTING)
        for part in self.particles:
            alpha = max(0, part.life / part.max_life)
            glPointSize(part.size * alpha + 1.0)
            glColor4f(part.color[0], part.color[1], part.color[2], alpha)
            glBegin(GL_POINTS)
            glVertex3f(*part.pos)
            glEnd()

    # ── Manos ─────────────────────────────────────────────────────────────
    def _render_hands(self):
        lm_l = self.tracker.landmarks_left
        lm_r = self.tracker.landmarks_right
        if lm_l: self._draw_hand_skeleton(lm_l, (0.3, 0.5, 1.0), _HAND_OFFSET_L)
        if lm_r: self._draw_hand_skeleton(lm_r, (0.2, 1.0, 0.5), _HAND_OFFSET_R)

    def _draw_hand_skeleton(self, landmarks, color, offset):
        """
        Renderiza esqueleto de mano.
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
        Direccion de la bala basada en la orientacion del dedo indice.
        Usa el vector MCP->TIP para determinar la direccion del disparo.
        """
        cam  = self.camera
        fwd  = cam.forward
        rgt  = cam.right
        lm_r = self.tracker.landmarks_right

        if lm_r and len(lm_r) > INDEX_TIP:
            tip = lm_r[INDEX_TIP]
            mcp = lm_r[INDEX_MCP]

            # Vector del dedo: MCP -> TIP en coordenadas de imagen
            dx = tip[0] - mcp[0]   # >0 = derecha fisica
            dy = tip[1] - mcp[1]   # <0 = arriba fisico

            # Componente lateral: cuanto se desvia el dedo a los lados
            AMP_LATERAL = 2.0
            # Componente vertical: cuanto apunta arriba/abajo
            AMP_VERTICAL = 4.0

            # Direccion: forward de la camara + desviacion lateral + vertical
            direction = [
                fwd[0] + rgt[0] * dx * AMP_LATERAL,
                (-dy) * AMP_VERTICAL,
                fwd[2] + rgt[2] * dx * AMP_LATERAL,
            ]
            # Normalizar
            mag = math.sqrt(sum(c*c for c in direction))
            if mag > 1e-6:
                direction = [c / mag for c in direction]
            else:
                direction = list(fwd)

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

        self._draw_hud_panels()
        if self.health <= 0:
            self._draw_game_over()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    # ── Panel glassmorphic ────────────────────────────────────────────────
    def _draw_panel(self, x, y, w, h, alpha=0.25,
                    border_color=(0.4, 0.8, 1.0), border_alpha=0.35):
        """Dibuja un panel semi-transparente con borde brillante."""
        # Fondo
        glColor4f(0.05, 0.08, 0.15, alpha)
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x+w, y)
        glVertex2f(x+w, y+h); glVertex2f(x, y+h)
        glEnd()
        # Borde superior (línea de acento)
        glColor4f(border_color[0], border_color[1], border_color[2], border_alpha)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex2f(x, y); glVertex2f(x+w, y)     # top
        glVertex2f(x, y+h); glVertex2f(x+w, y+h) # bottom
        glVertex2f(x, y); glVertex2f(x, y+h)     # left
        glVertex2f(x+w, y); glVertex2f(x+w, y+h) # right
        glEnd()
        # Línea superior brillante (acento)
        glColor4f(border_color[0], border_color[1], border_color[2], 0.6)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex2f(x+1, y); glVertex2f(x+w-1, y)
        glEnd()

    def _draw_bar(self, x, y, w, h, pct,
                  bg_color=(0.1, 0.1, 0.15), fill_colors=None):
        """Dibuja una barra de progreso con gradiente."""
        # Background
        glColor4f(bg_color[0], bg_color[1], bg_color[2], 0.5)
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x+w, y)
        glVertex2f(x+w, y+h); glVertex2f(x, y+h)
        glEnd()
        # Fill
        fill_w = max(0, w * max(0, min(1, pct)))
        if fill_w > 0:
            if fill_colors is None:
                # Gradiente verde → amarillo → rojo según porcentaje
                if pct > 0.6:
                    c = (0.15, 0.85, 0.6)
                elif pct > 0.3:
                    c = (0.9, 0.7, 0.15)
                else:
                    c = (0.95, 0.2, 0.15)
            else:
                c = fill_colors
            # Barra con ligero gradiente vertical
            glBegin(GL_QUADS)
            glColor4f(c[0]*0.7, c[1]*0.7, c[2]*0.7, 0.9)
            glVertex2f(x, y)
            glColor4f(c[0]*0.7, c[1]*0.7, c[2]*0.7, 0.9)
            glVertex2f(x+fill_w, y)
            glColor4f(c[0], c[1], c[2], 0.95)
            glVertex2f(x+fill_w, y+h)
            glColor4f(c[0], c[1], c[2], 0.95)
            glVertex2f(x, y+h)
            glEnd()
        # Borde de la barra
        glColor4f(0.4, 0.6, 0.8, 0.3)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(x, y); glVertex2f(x+w, y)
        glVertex2f(x+w, y); glVertex2f(x+w, y+h)
        glVertex2f(x+w, y+h); glVertex2f(x, y+h)
        glVertex2f(x, y+h); glVertex2f(x, y)
        glEnd()

    def _draw_hud_panels(self):
        elapsed = int(time.time() - self._start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        time_str = f"{minutes:02d}:{seconds:02d}"
        shield_pct = max(0, self.health) / 100.0

        # ── Panel principal (esquina superior izquierda) ──────────────────
        PX, PY = 16, 16
        PW, PH = 240, 130
        self._draw_panel(PX, PY, PW, PH, alpha=0.30,
                         border_color=(0.3, 0.7, 1.0))

        # PUNTOS
        lbl_score = self._font_label.render("PUNTOS", True, (120, 160, 200))
        self._blit_surface(lbl_score, (PX + 14, PY + 10))
        val_score = self._font_big.render(f"{self.score:,}", True, (80, 240, 180))
        self._blit_surface(val_score, (PX + 14, PY + 26))

        # Separador
        glColor4f(0.3, 0.6, 0.9, 0.2)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(PX + 14, PY + 58)
        glVertex2f(PX + PW - 14, PY + 58)
        glEnd()

        # ESCUDO label + porcentaje
        lbl_shield = self._font_label.render("ESCUDO", True, (120, 160, 200))
        self._blit_surface(lbl_shield, (PX + 14, PY + 64))
        # Color del porcentaje segun el nivel
        if shield_pct > 0.6:
            s_color = (80, 220, 160)
        elif shield_pct > 0.3:
            s_color = (230, 180, 50)
        else:
            s_color = (240, 70, 60)
        val_shield = self._font_medium.render(f"{max(0,self.health)}%", True, s_color)
        self._blit_surface(val_shield, (PX + PW - 14 - val_shield.get_width(), PY + 62))
        # Barra de escudo
        self._draw_bar(PX + 14, PY + 84, PW - 28, 10, shield_pct)

        # TIEMPO
        lbl_time = self._font_label.render("TIEMPO", True, (120, 160, 200))
        self._blit_surface(lbl_time, (PX + 14, PY + 100))
        val_time = self._font_medium.render(time_str, True, (180, 175, 255))
        self._blit_surface(val_time, (PX + PW - 14 - val_time.get_width(), PY + 98))

        # -- Indicador DISPARO (esquina superior derecha) --
        if self.tracker.shoot:
            fw, fh = 120, 36
            fx = self.width - fw - 16
            fy = 16
            # Panel highlight
            glColor4f(1.0, 0.85, 0.2, 0.15)
            glBegin(GL_QUADS)
            glVertex2f(fx, fy); glVertex2f(fx+fw, fy)
            glVertex2f(fx+fw, fy+fh); glVertex2f(fx, fy+fh)
            glEnd()
            # Borde dorado
            glColor4f(1.0, 0.85, 0.2, 0.6)
            glLineWidth(1.5)
            glBegin(GL_LINES)
            glVertex2f(fx, fy); glVertex2f(fx+fw, fy)
            glVertex2f(fx+fw, fy); glVertex2f(fx+fw, fy+fh)
            glVertex2f(fx+fw, fy+fh); glVertex2f(fx, fy+fh)
            glVertex2f(fx, fy+fh); glVertex2f(fx, fy)
            glEnd()
            fire_txt = self._font_medium.render("DISPARO", True, (255, 230, 80))
            self._blit_surface(fire_txt, (
                fx + (fw - fire_txt.get_width()) // 2,
                fy + (fh - fire_txt.get_height()) // 2,
            ))

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

    # ── Menú Principal ───────────────────────────────────────────────
    def _render_menu_bg(self):
        """Renderiza solo el starfield como fondo del menú."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, self.width / self.height, 0.05, 200.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.camera.apply()
        self._render_starfield()

    def _render_menu(self):
        """Dibuja el menú principal con título, botones e instrucciones."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cx = self.width // 2

        # ── Overlay sutil ──
        glColor4f(0.0, 0.0, 0.04, 0.55)
        glBegin(GL_QUADS)
        glVertex2f(0, 0); glVertex2f(self.width, 0)
        glVertex2f(self.width, self.height); glVertex2f(0, self.height)
        glEnd()

        # -- Titulo --
        title = self._font_hero.render("AR SPACE SHOOTER", True, (100, 200, 255))
        self._blit_surface(title, (cx - title.get_width() // 2, 100))

        # Separador decorativo bajo titulo
        sep_w = title.get_width() + 40
        glColor4f(0.3, 0.7, 1.0, 0.4)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex2f(cx - sep_w // 2, 165)
        glVertex2f(cx + sep_w // 2, 165)
        glEnd()

        # Subtitulo
        sub = self._font_small.render("Defiendete del campo de asteroides", True, (140, 160, 190))
        self._blit_surface(sub, (cx - sub.get_width() // 2, 175))

        # -- Botones --
        btn_w, btn_h = 220, 48
        btn_x = cx - btn_w // 2
        start_y = self.height // 2 + 10
        exit_y  = start_y + btn_h + 16

        self._draw_menu_button(btn_x, start_y, btn_w, btn_h,
                               "INICIAR", self._menu_hover == "start",
                               accent=(0.15, 0.85, 0.5))
        self._draw_menu_button(btn_x, exit_y, btn_w, btn_h,
                               "SALIR", self._menu_hover == "exit",
                               accent=(0.9, 0.3, 0.3))

        # ── Panel de instrucciones ──
        self._draw_instructions_panel()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_menu_button(self, x, y, w, h, text, hovered, accent=(0.4, 0.8, 1.0)):
        """Dibuja un botón del menú con efecto hover."""
        if hovered:
            # Fondo brillante al hover
            glColor4f(accent[0] * 0.3, accent[1] * 0.3, accent[2] * 0.3, 0.35)
        else:
            glColor4f(0.06, 0.08, 0.14, 0.45)
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x+w, y)
        glVertex2f(x+w, y+h); glVertex2f(x, y+h)
        glEnd()

        # Borde
        alpha = 0.7 if hovered else 0.35
        glColor4f(accent[0], accent[1], accent[2], alpha)
        glLineWidth(1.5 if not hovered else 2.0)
        glBegin(GL_LINES)
        glVertex2f(x, y); glVertex2f(x+w, y)
        glVertex2f(x+w, y); glVertex2f(x+w, y+h)
        glVertex2f(x+w, y+h); glVertex2f(x, y+h)
        glVertex2f(x, y+h); glVertex2f(x, y)
        glEnd()

        # Línea de acento superior
        glColor4f(accent[0], accent[1], accent[2], 0.8 if hovered else 0.4)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex2f(x+1, y); glVertex2f(x+w-1, y)
        glEnd()

        # Texto
        color = (int(accent[0]*255), int(accent[1]*255), int(accent[2]*255)) if hovered \
            else (200, 210, 220)
        txt = self._font_big.render(text, True, color)
        self._blit_surface(txt, (
            x + (w - txt.get_width()) // 2,
            y + (h - txt.get_height()) // 2,
        ))

    def _draw_instructions_panel(self):
        """Panel lateral con instrucciones de como jugar."""
        # Panel en la esquina inferior izquierda
        pw, ph = 380, 240
        px = 24
        py = self.height - ph - 24
        self._draw_panel(px, py, pw, ph, alpha=0.30,
                         border_color=(0.5, 0.7, 0.9), border_alpha=0.25)

        # Titulo
        title = self._font_medium.render("COMO JUGAR", True, (140, 190, 255))
        self._blit_surface(title, (px + 14, py + 10))

        # Separador
        glColor4f(0.4, 0.6, 0.9, 0.2)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(px + 14, py + 36)
        glVertex2f(px + pw - 14, py + 36)
        glEnd()

        # Instrucciones
        instructions = [
            ("MANO DERECHA  (disparar)",           (120, 200, 140)),
            ("  Extiende el dedo indice",           (170, 190, 175)),
            ("  Haz un movimiento rapido hacia",    (170, 190, 175)),
            ("  arriba (flick) para disparar",      (170, 190, 175)),
            ("  Apunta con la direccion del dedo",  (170, 190, 175)),
            ("",              (0, 0, 0)),
            ("MANO IZQUIERDA  (girar camara)",      (140, 170, 240)),
            ("  Mueve la mano a la izquierda",      (170, 175, 200)),
            ("  o derecha para rotar la vista",     (170, 175, 200)),
        ]

        y_off = py + 43
        for text, color in instructions:
            if text == "":
                y_off += 5
                continue
            surf = self._font_tiny.render(text, True, color)
            self._blit_surface(surf, (px + 14, y_off))
            y_off += 17

    def _draw_game_over(self):
        # Overlay oscuro
        glColor4f(0.0, 0.0, 0.02, 0.85)
        glBegin(GL_QUADS)
        glVertex2f(0, 0); glVertex2f(self.width, 0)
        glVertex2f(self.width, self.height); glVertex2f(0, self.height)
        glEnd()

        cx, cy = self.width // 2, self.height // 2

        # Panel central
        pw, ph = 380, 240
        px, py = cx - pw // 2, cy - ph // 2
        self._draw_panel(px, py, pw, ph, alpha=0.45,
                         border_color=(1.0, 0.3, 0.3), border_alpha=0.5)

        # FIN DEL JUEGO
        s1 = self._font_title.render("FIN DEL JUEGO", True, (255, 75, 75))
        self._blit_surface(s1, (cx - s1.get_width() // 2, py + 18))

        # Separador
        glColor4f(1.0, 0.3, 0.3, 0.3)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(px + 30, py + 68)
        glVertex2f(px + pw - 30, py + 68)
        glEnd()

        # Puntuacion final
        lbl = self._font_label.render("PUNTUACION FINAL", True, (180, 160, 160))
        self._blit_surface(lbl, (cx - lbl.get_width() // 2, py + 78))
        s2 = self._font_title.render(f"{self.score:,}", True, (255, 220, 80))
        self._blit_surface(s2, (cx - s2.get_width() // 2, py + 96))

        # Tiempo
        elapsed = int(time.time() - self._start_time)
        time_txt = self._font_small.render(
            f"Sobreviviste {elapsed // 60}m {elapsed % 60}s", True, (160, 160, 180))
        self._blit_surface(time_txt, (cx - time_txt.get_width() // 2, py + 145))

        # Controles
        s3 = self._font_small.render(
            "R = Reiniciar | M = Menu | ESC = Salir", True, (140, 140, 160))
        self._blit_surface(s3, (cx - s3.get_width() // 2, py + ph - 34))

    # ── OpenGL setup ──────────────────────────────────────────────────────
    def _setup_gl(self):
        glClearColor(0.02, 0.02, 0.06, 1.0)   # negro profundo espacial
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.15, 0.15, 0.20, 1.0))

    def _setup_light(self):
        glEnable(GL_LIGHT0)
        # Luz direccional fría (tipo sol distante)
        glLightfv(GL_LIGHT0, GL_POSITION, (10.0, 15.0, 10.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.90, 0.88, 0.95, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0,  1.0,  1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.12, 0.12, 0.18, 1.0))
