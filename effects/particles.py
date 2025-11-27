import numpy as np
import cv2


class Particle:
    def __init__(self, pos, vel, life=1.0, color=(180, 200, 255)):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.life = life
        self.age = 0.0
        self.color = color

    def update(self, dt):
        self.age += dt
        self.pos += self.vel * dt
        # simple drag
        self.vel *= (1.0 - 0.8 * dt)

    def alive(self):
        return self.age < self.life


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, pos, vel=(0, 0), count=4):
        for i in range(count):
            v = np.array(vel, dtype=float) + np.random.randn(2) * 30.0
            p = Particle(pos + np.random.randn(2) * 6.0, v, life=0.8 + np.random.rand() * 0.6)
            self.particles.append(p)

    def burst(self, pos, strength=1.0, count=30):
        for i in range(count):
            ang = np.random.rand() * 2.0 * np.pi
            speed = np.random.rand() * 200.0 * strength
            v = np.array([np.cos(ang) * speed, np.sin(ang) * speed])
            p = Particle(pos + np.random.randn(2) * 4.0, v, life=0.6 + np.random.rand() * 0.6)
            self.particles.append(p)

    def update(self, dt):
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive()]

    def draw(self, frame):
        # draw additive particles with glow
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for p in self.particles:
            alpha = max(0.0, 1.0 - (p.age / p.life))
            r = int(2 + 6 * alpha)
            # bluish neon color with variance
            col = (int(80 + 175 * alpha), int(140 + 115 * alpha), int(220 + 35 * alpha))
            cv2.circle(overlay, (int(p.pos[0]), int(p.pos[1])), r, col, -1)
        # add blur for glow
        blur = cv2.GaussianBlur(overlay, (0, 0), sigmaX=9, sigmaY=9)
        cv2.addWeighted(blur, 0.6, frame, 0.4, 0, frame)
