import math
import numpy as np
import cv2


class Shockwave:
    def __init__(self, center, lifespan=0.9):
        self.center = np.array(center, dtype=float)
        self.age = 0.0
        self.lifespan = lifespan

    def update(self, dt):
        self.age += dt

    def alive(self):
        return self.age < self.lifespan

    def radius(self):
        return (self.age / self.lifespan) * 400.0

    def alpha(self):
        t = self.age / self.lifespan
        return max(0.0, 1.0 - t)


class ShockwaveManager:
    def __init__(self):
        self.waves = []

    def trigger(self, center):
        self.waves.append(Shockwave(center))

    def update(self, dt):
        for w in self.waves:
            w.update(dt)
        self.waves = [w for w in self.waves if w.alive()]

    def draw(self, frame):
        h, w = frame.shape[:2]
        for s in self.waves:
            r = int(s.radius())
            a = s.alpha()
            # draw multiple concentric neon rings
            col = (200, 120, 255)
            for i in range(3):
                rr = int(r * (0.6 + 0.2 * i))
                thickness = max(1, int(6 * a * (1.0 - 0.25 * i)))
                cv2.circle(frame, (int(s.center[0]), int(s.center[1])), rr, col, thickness)
            # soft radial glow by drawing translucent filled circle
            glow = frame.copy()
            cv2.circle(glow, (int(s.center[0]), int(s.center[1])), max(10, int(r*0.4)), (200,120,255), -1)
            blur = cv2.GaussianBlur(glow, (0,0), sigmaX=28, sigmaY=28)
            cv2.addWeighted(blur, 0.25 * a, frame, 1.0, 0, frame)
