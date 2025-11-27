import numpy as np
import time


class PhysicsObject:
    def __init__(self, position, radius=40.0, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.radius = float(radius)
        self.mass = float(max(0.001, mass))
        self.forces = []
        self.damping = 0.9
        self.crushing = False
        self.crush_timer = 0.0
        self.crush_duration = 0.0
        self.base_radius = float(radius)
        self.energy_glitch = 0.0

    def apply_force(self, f):
        # f is 2D force
        self.forces.append(np.array(f, dtype=float))

    def apply_impulse(self, imp):
        # instantaneous change in momentum
        self.velocity += np.array(imp, dtype=float) / self.mass

    def crush(self, scale_factor, duration=0.5):
        self.crushing = True
        self.crush_timer = 0.0
        self.crush_duration = max(0.01, duration)
        self.target_scale = max(0.1, scale_factor)

    def update(self, dt):
        # integrate forces
        if len(self.forces) > 0:
            total = np.sum(self.forces, axis=0)
            self.forces = []
            acc = total / self.mass
            self.velocity += acc * dt

        # damping
        self.velocity *= (1.0 - 0.6 * dt)

        # integrate position
        self.position += self.velocity * dt

        # crush animation
        if self.crushing:
            self.crush_timer += dt
            t = min(1.0, self.crush_timer / self.crush_duration)
            # ease
            s = 1.0 - (1.0 - t) * (1.0 - t)
            self.radius = self.base_radius * (1.0 - s * (1.0 - self.target_scale))
            # glitch energy decays after crush finishes
            if self.crush_timer >= self.crush_duration:
                self.crushing = False
                self.radius = self.base_radius * self.target_scale

        # energy glitch decay
        self.energy_glitch = max(0.0, self.energy_glitch - dt * 0.8)
