import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


class HandTracker:
    def __init__(self, max_num_hands=1, detection_confidence=0.6, tracking_confidence=0.6):
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=max_num_hands,
                                     min_detection_confidence=detection_confidence,
                                     min_tracking_confidence=tracking_confidence)

    def process(self, frame):
        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res = self.hands.process(img)
        img.flags.writeable = True
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0]
        points = []
        for p in lm.landmark:
            points.append((int(p.x * w), int(p.y * h), p.z))

        # palm center as average of some landmarks
        inds = [0, 1, 5, 9, 13, 17]
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for i in inds:
            p = lm.landmark[i]
            cx += p.x
            cy += p.y
            cz += p.z
        cx /= len(inds)
        cy /= len(inds)
        cz /= len(inds)

        return {
            'points': [(int(x), int(y)) for x, y, _ in points],
            'landmarks': lm,
            'palm_center': np.array([cx * w, cy * h, cz], dtype=float),
            'raw': lm
        }
