import random
import math


class ScreenShaker:
    def __init__(self):
        self.time = 0.0
        self.duration = 0.0
        self.magnitude = 0.0

    def shake(self, magnitude=5.0, duration=0.5):
        self.magnitude = max(self.magnitude, magnitude)
        self.duration = max(self.duration, duration)
        self.time = 0.0

    def update(self, dt):
        if self.duration <= 0:
            self.magnitude = 0.0
            self.time = 0.0
            return
        self.time += dt
        if self.time >= self.duration:
            # decay
            self.magnitude = max(0.0, self.magnitude - dt * 8.0)
            self.duration = max(0.0, self.duration - dt)

    def get_offset(self):
        if self.magnitude <= 0.001:
            return (0, 0)
        return (random.uniform(-self.magnitude, self.magnitude), random.uniform(-self.magnitude, self.magnitude))
