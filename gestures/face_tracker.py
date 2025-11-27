import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection


class FaceTracker:
    def __init__(self, min_detection_confidence=0.5):
        self.detector = mp_face.FaceDetection(min_detection_confidence=min_detection_confidence)

    def process(self, frame):
        h, w = frame.shape[:2]
        img = frame.copy()
        img_rgb = mp.solutions.drawing_utils._normalized_to_pixel_coordinates
        # use mediapipe face detection
        import cv2
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.detector.process(img_rgb)
        if not res.detections:
            return None
        det = res.detections[0]
        loc = det.location_data
        if not loc or not loc.relative_bounding_box:
            return None
        rbb = loc.relative_bounding_box
        x = int(rbb.xmin * w)
        y = int(rbb.ymin * h)
        fw = int(rbb.width * w)
        fh = int(rbb.height * h)
        return {'bbox': (x, y, fw, fh), 'detection': det}
