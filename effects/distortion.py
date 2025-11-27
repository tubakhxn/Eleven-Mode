import numpy as np
import cv2


def apply_distortion(frame, strength=0.5):
    h, w = frame.shape[:2]
    # create displacement map based on noise + swirling effect
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    # use sinusoidal warp combined with random noise
    yy, xx = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    xx = xx.T
    yy = yy.T
    noise = (np.random.rand(h, w).astype(np.float32) - 0.5) * 2.0
    swirl = np.sin((xx - 0.5) * 6.28) * np.cos((yy - 0.5) * 6.28)
    disp_x = (np.sin(yy * 18.0) * 8.0 + noise * 14.0 + swirl * 12.0) * strength
    disp_y = (np.cos(xx * 18.0) * 8.0 + noise * 14.0 - swirl * 12.0) * strength
    for y in range(h):
        map_x[y, :] = np.arange(w, dtype=np.float32) + disp_x[y, :]
    for x in range(w):
        map_y[:, x] = np.arange(h, dtype=np.float32) + disp_y[:, x]

    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped
