"""
sounds.py
Generación procedural de efectos de sonido usando numpy + pygame.mixer.
No requiere archivos de audio externos.
"""

import numpy as np
import pygame


def _make_sound(samples_array: np.ndarray, sample_rate: int = 44100) -> pygame.mixer.Sound:
    """Convierte un array numpy float32 [-1,1] a un pygame.mixer.Sound."""
    # Convertir a int16
    samples_int = (samples_array * 32767).astype(np.int16)
    # Stereo: duplicar canal
    stereo = np.column_stack((samples_int, samples_int))
    return pygame.mixer.Sound(buffer=stereo.tobytes())


def generate_shoot_sound() -> pygame.mixer.Sound:
    """Sonido de disparo láser: barrido de frecuencia descendente."""
    sr = 44100
    duration = 0.15
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Barrido de 1800 Hz → 300 Hz
    freq = np.linspace(1800, 300, len(t))
    phase = np.cumsum(2 * np.pi * freq / sr)
    wave = np.sin(phase) * 0.5

    # Envolvente: ataque rápido, decay exponencial
    envelope = np.exp(-t * 25)
    wave *= envelope

    # Añadir un poco de ruido para textura
    noise = np.random.uniform(-0.1, 0.1, len(t))
    noise_env = np.exp(-t * 40)
    wave += noise * noise_env

    wave = np.clip(wave, -1, 1).astype(np.float32)
    return _make_sound(wave, sr)


def generate_explosion_sound() -> pygame.mixer.Sound:
    """Sonido de explosión: ruido filtrado con tono bajo."""
    sr = 44100
    duration = 0.45
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Componente de ruido (explosión)
    noise = np.random.uniform(-1, 1, len(t)).astype(np.float64)

    # Filtro simple: media móvil para bajar frecuencias
    kernel_size = 8
    kernel = np.ones(kernel_size) / kernel_size
    noise_filtered = np.convolve(noise, kernel, mode='same')

    # Tono bajo de impacto
    low_tone = np.sin(2 * np.pi * 60 * t) * 0.4
    low_tone2 = np.sin(2 * np.pi * 45 * t) * 0.3

    wave = noise_filtered * 0.5 + low_tone + low_tone2

    # Envolvente: ataque instantáneo, decay
    envelope = np.exp(-t * 6)
    wave *= envelope * 0.7

    wave = np.clip(wave, -1, 1).astype(np.float32)
    return _make_sound(wave, sr)


def generate_hit_sound() -> pygame.mixer.Sound:
    """Sonido de impacto metálico cuando un asteroide golpea al jugador."""
    sr = 44100
    duration = 0.3
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Tono metálico con armónicos
    wave = (np.sin(2 * np.pi * 220 * t) * 0.3 +
            np.sin(2 * np.pi * 440 * t) * 0.2 +
            np.sin(2 * np.pi * 880 * t) * 0.15 +
            np.sin(2 * np.pi * 1320 * t) * 0.1)

    # Ruido corto de impacto
    noise = np.random.uniform(-0.3, 0.3, len(t))
    noise_env = np.exp(-t * 50)
    wave += noise * noise_env

    # Envolvente
    envelope = np.exp(-t * 12)
    wave *= envelope

    wave = np.clip(wave, -1, 1).astype(np.float32)
    return _make_sound(wave, sr)


def generate_ambient_hum() -> pygame.mixer.Sound:
    """Zumbido ambiente de nave espacial — loop continuo."""
    sr = 44100
    duration = 3.0  # se loopea
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Zumbido bajo tipo reactor/motor
    wave = (np.sin(2 * np.pi * 55 * t) * 0.08 +
            np.sin(2 * np.pi * 82.5 * t) * 0.05 +
            np.sin(2 * np.pi * 110 * t) * 0.03)

    # Modulación lenta para que no sea estático
    mod = 1.0 + 0.15 * np.sin(2 * np.pi * 0.3 * t)
    wave *= mod

    # Ruido muy suave de fondo
    noise = np.random.uniform(-0.02, 0.02, len(t))
    wave += noise

    wave = np.clip(wave, -1, 1).astype(np.float32)
    return _make_sound(wave, sr)


class SoundManager:
    """Gestiona todos los efectos de sonido del juego."""

    def __init__(self):
        # Inicializar mixer si no lo está
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

        # Generar sonidos procedurales
        print("[SoundManager] Generando sonidos...")
        self.snd_shoot     = generate_shoot_sound()
        self.snd_explosion = generate_explosion_sound()
        self.snd_hit       = generate_hit_sound()
        self.snd_ambient   = generate_ambient_hum()

        # Volúmenes
        self.snd_shoot.set_volume(0.4)
        self.snd_explosion.set_volume(0.5)
        self.snd_hit.set_volume(0.6)
        self.snd_ambient.set_volume(0.15)

        self._ambient_playing = False
        print("[SoundManager] ✓ Sonidos generados")

    def play_shoot(self):
        self.snd_shoot.play()

    def play_explosion(self):
        self.snd_explosion.play()

    def play_hit(self):
        self.snd_hit.play()

    def start_ambient(self):
        if not self._ambient_playing:
            self.snd_ambient.play(loops=-1)  # loop infinito
            self._ambient_playing = True

    def stop_ambient(self):
        if self._ambient_playing:
            self.snd_ambient.stop()
            self._ambient_playing = False
