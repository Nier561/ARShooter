"""
asteroid.py
Asteroide procedural 3D: icosfera deformada con rotación.
Tres tamaños: small, medium, large.
"""

import math
import random
from OpenGL.GL import (
    glPushMatrix, glPopMatrix, glTranslatef, glRotatef,
    glMaterialfv, glBegin, glEnd, glVertex3f, glNormal3f,
    GL_FRONT, GL_AMBIENT_AND_DIFFUSE, GL_SPECULAR, GL_SHININESS,
    GL_EMISSION, GL_TRIANGLES,
)


# ── Parámetros por tamaño ────────────────────────────────────────────────
SIZE_PARAMS = {
    "small":  {"radius": 0.30, "speed": 2.2, "points": 50,  "subdivisions": 1},
    "medium": {"radius": 0.55, "speed": 1.4, "points": 100, "subdivisions": 2},
    "large":  {"radius": 0.85, "speed": 0.9, "points": 200, "subdivisions": 2},
}


def _normalize(v):
    mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if mag < 1e-8:
        return (0.0, 0.0, 1.0)
    return (v[0]/mag, v[1]/mag, v[2]/mag)


def _icosahedron():
    """Retorna (vértices, caras) de un icosaedro unitario."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        _normalize((-1,  t,  0)), _normalize(( 1,  t,  0)),
        _normalize((-1, -t,  0)), _normalize(( 1, -t,  0)),
        _normalize(( 0, -1,  t)), _normalize(( 0,  1,  t)),
        _normalize(( 0, -1, -t)), _normalize(( 0,  1, -t)),
        _normalize(( t,  0, -1)), _normalize(( t,  0,  1)),
        _normalize((-t,  0, -1)), _normalize((-t,  0,  1)),
    ]
    faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]
    return verts, faces


def _subdivide(verts, faces):
    """Subdivide cada triángulo en 4 triángulos (Loop subdivision simplificada)."""
    edge_cache = {}
    new_faces = []

    def midpoint(i, j):
        key = (min(i,j), max(i,j))
        if key in edge_cache:
            return edge_cache[key]
        v1, v2 = verts[i], verts[j]
        mid = _normalize(((v1[0]+v2[0])/2, (v1[1]+v2[1])/2, (v1[2]+v2[2])/2))
        idx = len(verts)
        verts.append(mid)
        edge_cache[key] = idx
        return idx

    for (a, b, c) in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.extend([
            (a, ab, ca),
            (b, bc, ab),
            (c, ca, bc),
            (ab, bc, ca),
        ])

    return verts, new_faces


def _generate_asteroid_mesh(radius, subdivisions, deform=0.3):
    """
    Genera una malla de asteroide: icosfera subdividida con vértices
    desplazados aleatoriamente para dar aspecto rocoso irregular.
    Retorna lista de triángulos [(v0,v1,v2,normal), ...].
    """
    verts, faces = _icosahedron()
    verts = list(verts)

    for _ in range(subdivisions):
        verts, faces = _subdivide(verts, faces)

    # Deformar radialmente
    deformed = []
    for v in verts:
        scale = radius * (1.0 + random.uniform(-deform, deform))
        deformed.append((v[0]*scale, v[1]*scale, v[2]*scale))

    # Calcular normales por cara y construir triángulos
    triangles = []
    for (a, b, c) in faces:
        v0, v1, v2 = deformed[a], deformed[b], deformed[c]
        # Normal del triángulo
        e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
        e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
        nx = e1[1]*e2[2] - e1[2]*e2[1]
        ny = e1[2]*e2[0] - e1[0]*e2[2]
        nz = e1[0]*e2[1] - e1[1]*e2[0]
        normal = _normalize((nx, ny, nz))
        triangles.append((v0, v1, v2, normal))

    return triangles


class Asteroid:
    MIN_DIST = 15.0
    MAX_DIST = 30.0

    # Colores rocosos con variación
    _BASE_COLORS = [
        (0.45, 0.38, 0.32),   # marrón grisáceo
        (0.50, 0.45, 0.40),   # arena oscura
        (0.38, 0.35, 0.38),   # gris púrpura
        (0.55, 0.48, 0.38),   # marrón claro
        (0.40, 0.40, 0.42),   # gris neutro
    ]

    def __init__(self, player_pos: list, cam_forward: list = None,
                 size_category: str = "medium"):
        params = SIZE_PARAMS.get(size_category, SIZE_PARAMS["medium"])
        self.size_category = size_category
        self.radius   = params["radius"]
        self.speed    = params["speed"]
        self.points   = params["points"]
        self.alive    = True

        # Posición
        self.pos = self._random_spawn(player_pos, cam_forward)

        # Rotación
        self.rot_axis  = _normalize((
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
        ))
        self.rot_angle = random.uniform(0, 360)
        self.rot_speed = random.uniform(30, 120)

        # Color
        base = random.choice(self._BASE_COLORS)
        var = 0.08
        self.color_diffuse = (
            max(0, min(1, base[0] + random.uniform(-var, var))),
            max(0, min(1, base[1] + random.uniform(-var, var))),
            max(0, min(1, base[2] + random.uniform(-var, var))),
            1.0,
        )
        self.color_specular = (0.3, 0.3, 0.3, 1.0)
        self.shininess = 10.0

        # Mesh procedural
        self._triangles = _generate_asteroid_mesh(
            self.radius, params["subdivisions"], deform=0.35
        )

    def update(self, dt: float, player_pos: list):
        dx = player_pos[0] - self.pos[0]
        dy = player_pos[1] - self.pos[1]
        dz = player_pos[2] - self.pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 0.01:
            return
        step = self.speed * dt
        self.pos[0] += (dx / dist) * step
        self.pos[1] += (dy / dist) * step
        self.pos[2] += (dz / dist) * step

        # Rotación continua
        self.rot_angle += self.rot_speed * dt
        if self.rot_angle > 360:
            self.rot_angle -= 360

        # Si llega al jugador → daño
        if dist < 0.5 + self.radius:
            self.alive = False

    def render(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glRotatef(self.rot_angle, self.rot_axis[0], self.rot_axis[1], self.rot_axis[2])

        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, self.color_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR,            self.color_specular)
        glMaterialfv(GL_FRONT, GL_SHININESS,           [self.shininess])
        glMaterialfv(GL_FRONT, GL_EMISSION,            (0.05, 0.03, 0.02, 1.0))

        glBegin(GL_TRIANGLES)
        for (v0, v1, v2, normal) in self._triangles:
            glNormal3f(*normal)
            glVertex3f(*v0)
            glVertex3f(*v1)
            glVertex3f(*v2)
        glEnd()

        glMaterialfv(GL_FRONT, GL_EMISSION, (0.0, 0.0, 0.0, 1.0))
        glPopMatrix()

    def check_collision(self, proj_pos: list) -> bool:
        dx = proj_pos[0] - self.pos[0]
        dy = proj_pos[1] - self.pos[1]
        dz = proj_pos[2] - self.pos[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz) < self.radius + 0.15

    @staticmethod
    def _random_spawn(player_pos: list, cam_forward: list = None) -> list:
        """Spawn en un arco frontal, con Y variada (flotando en el espacio)."""
        dist = random.uniform(Asteroid.MIN_DIST, Asteroid.MAX_DIST)

        if cam_forward is not None:
            base_angle = math.atan2(cam_forward[0], cam_forward[2])
        else:
            base_angle = math.pi

        spread = math.radians(45)
        angle  = base_angle + random.uniform(-spread, spread)
        y_off  = random.uniform(-1.5, 4.0)  # más rango vertical en el espacio

        return [
            player_pos[0] + dist * math.sin(angle),
            player_pos[1] + y_off,
            player_pos[2] + dist * math.cos(angle),
        ]
