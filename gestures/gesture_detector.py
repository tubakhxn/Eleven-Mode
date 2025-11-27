import numpy as np
import time

from collections import deque


def _dist(a, b):
    return np.linalg.norm(a - b)


class GestureDetector:
    def __init__(self):
        self.history = deque(maxlen=8)
        self.state = None
        self.cooldowns = {}
        self.last_time = time.time()

    def _finger_tips(self, lm):
        # returns numpy array positions for finger tips (thumb tip idx=4 etc.)
        pts = []
        for idx in (4, 8, 12, 16, 20):
            p = lm.landmark[idx]
            pts.append(np.array([p.x, p.y, p.z]))
        return pts

    def update(self, hand, dt, thresholds=None):
        lm = hand['raw']
        hpos = np.array([hand['palm_center'][0], hand['palm_center'][1]])
        tips = self._finger_tips(lm)

        # convert normalized tips to pixel by approximating with palm center scale
        tip_pts = [np.array([tip[0] * hand['palm_center'][0] / hand['palm_center'][0] * hand['palm_center'][0], tip[1] * hand['palm_center'][1] / hand['palm_center'][1]]) for tip in tips]

        # compute distances
        thumb = np.array([lm.landmark[4].x * hand['palm_center'][0], lm.landmark[4].y * hand['palm_center'][1]])
        index = np.array([lm.landmark[8].x * hand['palm_center'][0], lm.landmark[8].y * hand['palm_center'][1]])
        dist_thumb_index = np.linalg.norm(thumb - index)

        # palm open: average finger spread
        spread = 0.0
        for i in (8, 12, 16, 20):
            p = lm.landmark[i]
            v = np.array([p.x - lm.landmark[0].x, p.y - lm.landmark[0].y])
            spread += np.linalg.norm(v)
        spread /= 4.0

        # fist detection: tips close to wrist
        wrist = np.array([lm.landmark[0].x * hand['palm_center'][0], lm.landmark[0].y * hand['palm_center'][1]])
        fist_score = 0.0
        for i in (4, 8, 12, 16, 20):
            p = np.array([lm.landmark[i].x * hand['palm_center'][0], lm.landmark[i].y * hand['palm_center'][1]])
            d = np.linalg.norm(p - wrist)
            if d < 40:
                fist_score += 1

        # velocity estimate using history of palm centers
        pc = np.array([hand['palm_center'][0], hand['palm_center'][1], hand['palm_center'][2]])
        now = time.time()
        self.history.append((now, pc))
        vel = np.array([0.0, 0.0, 0.0])
        if len(self.history) >= 2:
            t0, p0 = self.history[0]
            t1, p1 = self.history[-1]
            dt_hist = max(1e-6, t1 - t0)
            vel = (p1 - p0) / dt_hist

        out = None

        # threshold values (defaults)
        if thresholds is None:
            thresholds = {}
        pinch_px_thresh = thresholds.get('pinch_px', 60)
        fist_count_thresh = thresholds.get('fist_count', 4)
        push_z_thresh = thresholds.get('push_z', -0.018)
        levitate_spread_thresh = thresholds.get('levitate_spread', 0.12)

        # pinch detection (pixel-proximity threshold)
        # map normalized thumb/index to pixels
        thumb_px = np.array([lm.landmark[4].x * hand['palm_center'][0], lm.landmark[4].y * hand['palm_center'][1]])
        index_px = np.array([lm.landmark[8].x * hand['palm_center'][0], lm.landmark[8].y * hand['palm_center'][1]])
        dist_thumb_index_px = np.linalg.norm(thumb_px - index_px)

        if dist_thumb_index_px < pinch_px_thresh:
            # map to pixel point mid
            pt = np.array([(lm.landmark[4].x + lm.landmark[8].x) / 2.0 * hand['palm_center'][0], (lm.landmark[4].y + lm.landmark[8].y) / 2.0 * hand['palm_center'][1]])
            # determine start/hold/end by state
            if self.state != 'pinch':
                self.state = 'pinch'
                out = {'type': 'pinch', 'state': 'start', 'point': pt}
            else:
                out = {'type': 'pinch', 'state': 'hold', 'point': pt}
        else:
            if self.state == 'pinch':
                self.state = None
                pt = np.array([lm.landmark[8].x * hand['palm_center'][0], lm.landmark[8].y * hand['palm_center'][1]])
                out = {'type': 'pinch', 'state': 'end', 'point': pt}

        # detect palm push: quick forward z velocity (negative z -> towards camera in mediapipe)
        if out is None:
            zvel = vel[2]
            if zvel < push_z_thresh:  # rapid push forward
                pt = np.array([hand['palm_center'][0], hand['palm_center'][1]])
                out = {'type': 'palm_push', 'state': 'trigger', 'point': pt, 'power': -zvel, 'normal': np.array([0.0, -1.0])}

        # slow open palm -> levitate (low velocity and high spread)
        if out is None:
            if spread > levitate_spread_thresh and np.linalg.norm(vel[:2]) < 8.0:
                pt = np.array([hand['palm_center'][0], hand['palm_center'][1]])
                out = {'type': 'levitate', 'state': 'hold', 'point': pt, 'power': spread}

        # fist -> trigger compress
        if out is None:
            if fist_score >= fist_count_thresh:
                pt = np.array([hand['palm_center'][0], hand['palm_center'][1]])
                out = {'type': 'fist', 'state': 'trigger', 'point': pt}

        if out is None:
            # idle default
            out = {'type': 'none', 'state': 'idle', 'point': np.array([hand['palm_center'][0], hand['palm_center'][1]])}

        return out
