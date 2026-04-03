"""
AR Space Shooter - main.py
Punto de entrada principal. Inicializa Pygame + OpenGL y arranca el bucle de juego.
"""

import sys
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE
from OpenGL.GL import glViewport
from game_world import GameWorld


def main():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    WIDTH, HEIGHT = 1280, 720
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("AR Space Shooter")
    pygame.mouse.set_visible(True)   # Visible en menú

    glViewport(0, 0, WIDTH, HEIGHT)

    clock = pygame.time.Clock()
    world = GameWorld(WIDTH, HEIGHT)
    world.start()          # Arranca hilo de visión artificial

    running = True
    while running:
        dt = clock.tick(60) / 1000.0   # delta-time en segundos

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False

            # Pasar eventos al GameWorld (menú, game over, etc.)
            result = world.handle_event(event)
            if result == "quit":
                running = False

        world.update(dt)
        world.render()
        pygame.display.flip()

    world.stop()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
