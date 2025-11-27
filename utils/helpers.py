import cv2
import numpy as np


def vec2(x, y):
    return np.array([float(x), float(y)], dtype=float)


def clamp(v, a, b):
    return max(a, min(b, v))


def smoothstep(a, b, t):
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    x = (t - a) / (b - a)
    return x * x * (3 - 2 * x)


def draw_neon_circle(frame, center, radius, color=(200, 60, 255), thickness=2):
    # draw base filled circle
    cv2.circle(frame, center, int(radius), color, thickness)
    # outer pulse
    overlay = frame.copy()
    cv2.circle(overlay, center, int(radius + 8), color, -1)
    alpha = 0.15
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # small inner highlight
    cv2.circle(frame, center, max(2, int(radius * 0.12)), (255, 255, 255), -1)
