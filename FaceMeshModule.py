import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=2, refine_landmarks=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.staticMode = static_image_mode
        self.maxFaces = max_num_faces
        self.refineLandmarks = refine_landmarks
        self.minDetectionCon = min_detection_confidence
        self.minTrackCon = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refineLandmarks,
                                                 self.minDetectionCon, self.minTrackCon)
        self.DrawSpecs = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.results = None


    def FindFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        landmarks = []

        if self.results.multi_face_landmarks:
            # 1. Loop through the results
            for facelms in self.results.multi_face_landmarks:
                # 2. If draw is True, draw the landmarks on the image
                if draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec=self.DrawSpecs)

                face_lm = []
                face_lamdmarks = []
                
                # 1. Get the height, width and channels of the image
                # 2. Get the x and y coordinates of the landmarks
                # 3. Append the x and y coordinates to the face_lm list
                # 4. Append the x, y and z coordinates to the face_lamdmarks list
                for id, lm in enumerate(facelms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face_lm.append([cx, cy])
                    face_lamdmarks.append([lm.x, lm.y, lm.z])

                # 1. Append the face_lm list to the faces list
                landmarks.append(face_lamdmarks)
                faces.append(face_lm)

        return img, faces, landmarks
