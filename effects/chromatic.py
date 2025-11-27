import cv2
import numpy as np


def chromatic_aberration(frame, intensity=8):
    # Shift color channels slightly to create chromatic aberration
    h, w = frame.shape[:2]
    b, g, r = cv2.split(frame)
    # offsets scaled by intensity
    dx = int(max(1, intensity))
    # create translation matrices
    M1 = np.float32([[1, 0, -dx], [0, 1, 0]])
    M2 = np.float32([[1, 0, dx//2], [0, 1, 0]])
    rb = cv2.warpAffine(b, M1, (w, h), borderMode=cv2.BORDER_REFLECT)
    gg = cv2.warpAffine(g, M2, (w, h), borderMode=cv2.BORDER_REFLECT)
    rr = r
    merged = cv2.merge([rb, gg, rr])
    # boost contrast and tint slightly
    out = cv2.convertScaleAbs(merged, alpha=1.05 + intensity*0.02, beta=6)
    return out


class FreezeEffect:
    def __init__(self):
        self.timer = 0.0
        self.duration = 0.0
        self.frame = None
        self.intensity = 8

    def trigger(self, frame, power=1.0, duration=0.8):
        self.frame = frame.copy()
        self.duration = max(0.05, duration)
        self.timer = self.duration
        self.intensity = int(min(40, 6 + power * 20))

    def update(self, dt):
        if self.timer > 0.0:
            self.timer -= dt

    def active(self):
        return self.timer > 0.0

    def render(self):
        if self.frame is None:
            return None
        # apply chromatic with intensity scaled by remaining time
        t = max(0.0, self.timer / self.duration)
        inten = int(self.intensity * (0.6 + 0.4 * t))
        out = chromatic_aberration(self.frame, inten)
        # desaturate slightly and add vignette
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, 0.85, gray3, 0.15, 0)
        return out
