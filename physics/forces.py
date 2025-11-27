import numpy as np


class Force:
    def __init__(self, vector):
        self.vector = np.array(vector, dtype=float)

    def apply(self, obj):
        obj.apply_force(self.vector)
