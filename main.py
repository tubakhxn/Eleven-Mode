import cv2
import time
import random
import numpy as np

from gestures.hand_tracking import HandTracker
from gestures.face_tracker import FaceTracker
from gestures.gesture_detector import GestureDetector
from physics.objects import PhysicsObject
from physics.forces import Force
from effects.particles import ParticleSystem
from effects.shockwave import ShockwaveManager
from effects.distortion import apply_distortion
from effects.chromatic import FreezeEffect
from effects.glow import draw_glow
from effects.screenshake import ScreenShaker
from utils.helpers import draw_neon_circle, vec2, clamp, smoothstep


def create_scene(n=6, w=1280, h=720):
    objs = []
    for i in range(n):
        pos = np.array([random.uniform(w*0.2, w*0.8), random.uniform(h*0.2, h*0.8)], dtype=float)
        r = random.uniform(30, 60)
        objs.append(PhysicsObject(position=pos, radius=r, mass=r*0.1))
    return objs


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = HandTracker(max_num_hands=1)
    detector = GestureDetector()
    face_tracker = FaceTracker(min_detection_confidence=0.6)
    freeze = FreezeEffect()

    objects = create_scene(6, w, h)

    particle_sys = ParticleSystem()
    shock_mgr = ShockwaveManager()
    shaker = ScreenShaker()

    last = time.time()

    grabbed = None
    grab_offset = np.zeros(2)

    # motion accumulation buffer to create trail and stronger FX compositing
    accum = np.zeros((h, w, 3), dtype=np.float32)

    # create sliders for tuning
    cv2.namedWindow('Mind-Power Telekinesis UI', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('pinch_px', 'Mind-Power Telekinesis UI', 60, 200, lambda x: None)
    cv2.createTrackbar('fist_count', 'Mind-Power Telekinesis UI', 4, 5, lambda x: None)
    cv2.createTrackbar('push_z_thresh', 'Mind-Power Telekinesis UI', 18, 100, lambda x: None)
    cv2.createTrackbar('levitate_spread', 'Mind-Power Telekinesis UI', 12, 50, lambda x: None)
    cv2.createTrackbar('push_power_trigger', 'Mind-Power Telekinesis UI', 6, 50, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        dt = np.clip(now - last, 1e-3, 0.05)
        last = now

        frame = cv2.flip(frame, 1)
        canvas = frame.copy()
        # darken background slightly so effects pop
        dark = cv2.convertScaleAbs(frame, alpha=0.55, beta=-30)
        canvas = dark

        hand = tracker.process(frame)
        face = face_tracker.process(frame)
        gesture = None
        # read thresholds from sliders and build threshold dict
        pinch_px = cv2.getTrackbarPos('pinch_px', 'Mind-Power Telekinesis UI')
        fist_count = cv2.getTrackbarPos('fist_count', 'Mind-Power Telekinesis UI')
        push_z_raw = cv2.getTrackbarPos('push_z_thresh', 'Mind-Power Telekinesis UI')
        # slider maps 0..100 to z threshold -0.005 .. -0.05 roughly
        push_z = - (0.005 + (push_z_raw / 100.0) * 0.045)
        lev_spread_raw = cv2.getTrackbarPos('levitate_spread', 'Mind-Power Telekinesis UI')
        lev_spread = lev_spread_raw / 100.0
        push_power_trigger = cv2.getTrackbarPos('push_power_trigger', 'Mind-Power Telekinesis UI') / 10.0

        thresholds = {'pinch_px': pinch_px, 'fist_count': fist_count, 'push_z': push_z, 'levitate_spread': lev_spread}

        if hand is not None:
            gesture = detector.update(hand, dt, thresholds)

        # Object interaction logic
        if gesture is not None and gesture['type'] == 'pinch' and gesture['state'] == 'start':
            # select closest object to pinch point
            p = gesture['point']
            best = None
            bestd = 1e9
            for obj in objects:
                d = np.linalg.norm(obj.position - p)
                if d < bestd and d < 120:
                    bestd = d
                    best = obj
            if best is not None:
                grabbed = best
                grab_offset = grabbed.position - p
                # pinch burst and glow
                particle_sys.burst(grabbed.position, strength=0.6, count=18)
                grabbed.energy_glitch = min(1.0, grabbed.energy_glitch + 0.2)

        if gesture is not None and gesture['type'] == 'pinch' and gesture['state'] == 'end':
            grabbed = None

        # pinch hold moves object
        if grabbed is not None and gesture is not None and gesture['type'] == 'pinch' and gesture['state'] == 'hold':
            target = gesture['point'] + grab_offset
            # apply spring-like force to follow
            dirv = target - grabbed.position
            grabbed.apply_force(dirv * 25.0)
            # particle trail
            particle_sys.emit(grabbed.position, vel=-grabbed.velocity*0.05, count=2)

        # palm push -> telekinetic throw (impulse away from hand normal)
        if gesture is not None and gesture['type'] == 'palm_push' and gesture['state'] == 'trigger':
            p = gesture['point']
            normal = gesture.get('normal', np.array([0.0, -1.0]))
            shock_mgr.trigger(p)
            shaker.shake(0.35, 0.6)
            # radial particle burst
            particle_sys.burst(p, strength=min(2.5, 1.0 + gesture.get('power', 0.8)), count=40)
            # trigger freeze/chromatic if power exceeds slider
            power_val = gesture.get('power', 0.8)
            if power_val > push_power_trigger:
                freeze.trigger(frame, power=power_val, duration=0.8)
            for obj in objects:
                to_obj = obj.position - p
                dist = max(np.linalg.norm(to_obj), 1.0)
                direction = to_obj / dist
                impulse = direction * (6000.0 / (dist + 50.0))
                obj.apply_impulse(impulse)
                obj.energy_glitch = min(1.0, obj.energy_glitch + 0.8 + min(0.6, gesture.get('power', 0.8)))

        # slow open palm -> levitate
        if gesture is not None and gesture['type'] == 'levitate' and gesture['state'] == 'hold':
            p = gesture['point']
            # gentle upward force to nearby objects
            for obj in objects:
                d = np.linalg.norm(obj.position - p)
                if d < 200:
                    obj.apply_force(np.array([0.0, -200.0 * (1.0 - d/200.0)]))
                    particle_sys.emit(obj.position, vel=np.array([0.0, -0.5]), count=1)

        # fist -> compress/crush
        if gesture is not None and gesture['type'] == 'fist' and gesture['state'] == 'trigger':
            p = gesture['point']
            shaker.shake(0.5, 1.0)
            for obj in objects:
                if np.linalg.norm(obj.position - p) < 200:
                    obj.crush(0.35, duration=0.6)
                    obj.energy_glitch = min(1.0, obj.energy_glitch + 1.0)
                    # heavy radial distortion and particles
                    particle_sys.burst(obj.position, strength=1.5, count=60)

        # update physics
        for obj in objects:
            obj.update(dt)
            # keep inside screen
            obj.position[0] = clamp(obj.position[0], obj.radius, w - obj.radius)
            obj.position[1] = clamp(obj.position[1], obj.radius, h - obj.radius)

        # update effects
        particle_sys.update(dt)
        shock_mgr.update(dt)
        shaker.update(dt)

        # render
        # draw everything onto an overlay then composite with original frame
        overlay = canvas.copy()
        # apply screen shake offset
        ofs = shaker.get_offset()

        # draw objects into overlay
        for obj in objects:
            # distortion if glitching
            if obj.energy_glitch > 0.01 and obj.crushing:
                # render object to mask then distort
                tmp = canvas.copy()
                draw_neon_circle(tmp, tuple(obj.position.astype(int)), int(obj.radius), (200, 60, 255), thickness=-1)
                tmp = apply_distortion(tmp, obj.energy_glitch)
                # composite
                alpha = min(1.0, obj.energy_glitch)
                cv2.addWeighted(tmp, alpha, canvas, 1.0 - alpha, 0, canvas)
            else:
                draw_neon_circle(canvas, tuple(obj.position.astype(int)), int(obj.radius), (200, 60, 255), thickness=-1)

            # glow outline (on overlay)
            draw_glow(overlay, obj.position.astype(int), int(obj.radius))

            # particle overlay per-object
            particle_sys.draw(overlay)

            # if grabbed and this is grabbed object, draw telekinesis beam
            if grabbed is not None and obj is grabbed and gesture is not None and gesture['type'] == 'pinch' and gesture['state'] == 'hold':
                p = gesture['point']
                pt1 = (int(p[0]), int(p[1]))
                pt2 = (int(obj.position[0]), int(obj.position[1]))
                # bright beam
                cv2.line(overlay, pt1, pt2, (200, 220, 255), 6)
                # inner bright core
                cv2.line(overlay, pt1, pt2, (255, 255, 255), 2)
                # add small particles along beam
                for i in range(6):
                    t = i / 6.0
                    pos = obj.position * t + p * (1 - t)
                    particle_sys.emit(pos, vel=np.random.randn(2) * 10.0, count=1)

        # shockwaves
        shock_mgr.draw(overlay)

        # face guidance: if face detected, make it brighter and draw a soft outline
        if face is not None:
            x, y, fw, fh = face['bbox']
            # brighten face area slightly so it's visible under effects
            rx1, ry1 = max(0, x - 10), max(0, y - 10)
            rx2, ry2 = min(w, x + fw + 10), min(h, y + fh + 10)
            face_roi = frame[ry1:ry2, rx1:rx2]
            if face_roi.size != 0:
                # gently brighten
                bright = cv2.convertScaleAbs(face_roi, alpha=1.25, beta=20)
                overlay[ry1:ry2, rx1:rx2] = bright
            # neon rectangle
            cv2.rectangle(overlay, (x, y), (x + fw, y + fh), (180, 220, 255), 2)

        # draw hand debug and neon rings
        if hand is not None:
            # draw landmark points
            for (x, y) in hand['points']:
                cv2.circle(canvas, (int(x), int(y)), 4, (255, 200, 50), -1)
            # power ring
            if gesture is not None and gesture['type'] in ('palm_push', 'levitate'):
                r = int(60 + 40 * abs(gesture.get('power', 1.0)))
                cv2.circle(canvas, tuple(gesture['point'].astype(int)), r, (200, 120, 255), 2)

        # update freeze effect
        freeze.update(dt)
        if freeze.active():
            # use freeze-rendered chromatic frame as main output during freeze
            frozen = freeze.render()
            if frozen is not None:
                # blend frozen over the frame for intensity
                composed = cv2.addWeighted(frozen, 0.9, frame, 0.1, 0)
            else:
                composed = frame.copy()
        else:
            # accumulate overlay into motion buffer for stronger trails and smoother blending
            # convert overlay to float
            fov = overlay.astype(np.float32)
            accum = accum * 0.82 + fov * 0.18
            # composite: keep face bright by blending original frame and accumulated overlay
            composed = cv2.addWeighted(frame, 0.40, cv2.convertScaleAbs(accum), 0.60, 0)

        # draw on-screen control bar at top showing slider names and values
        ui = composed.copy()
        bar_h = 56
        cv2.rectangle(ui, (0, 0), (w, bar_h), (10, 10, 12), -1)
        # semi-transparent overlay
        cv2.addWeighted(ui, 0.85, composed, 0.15, 0, ui)
        # draw slider labels and values
        labels = [
            (f'pinch_px: {pinch_px}', 20),
            (f'fist_count: {fist_count}', 220),
            (f'push_z: {push_z:.3f}', 420),
            (f'lev_spread: {lev_spread:.2f}', 640),
            (f'push_power_trig: {push_power_trigger:.1f}', 860),
        ]
        for text, x in labels:
            cv2.putText(ui, text, (x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 235, 255), 2, cv2.LINE_AA)

        # apply final screen shake transform
        M = np.float32([[1, 0, ofs[0]], [0, 1, ofs[1]]])
        shaken = cv2.warpAffine(ui, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # show gesture label
        if gesture is not None and gesture['type'] != 'none':
            cv2.putText(shaken, f"Gesture: {gesture['type']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 220, 255), 2)

        # show
        cv2.imshow('Mind-Power Telekinesis UI', shaken)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
