import cv2
import numpy as np


def draw_glow(frame, center, radius):
    # draw neon glowing outline using blurred mask
    mask = np.zeros_like(frame)
    cv2.circle(mask, tuple(center), int(radius + 6), (255, 255, 255), -1)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=15, sigmaY=15)
    # normalize
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    glow = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_OCEAN)
    # overlay
    alpha = 0.45
    cv2.addWeighted(glow, alpha, frame, 1 - alpha, 0, frame)
