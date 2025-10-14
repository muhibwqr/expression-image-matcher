import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

mp_face_mesh = mp.solutions.face_mesh

def extract_face_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        points = []
        for lm in landmarks.landmark:
            points.extend([lm.x, lm.y, lm.z])
        return np.array(points)

def compare_expressions(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return 0
    landmarks1 = landmarks1.reshape(1, -1)
    landmarks2 = landmarks2.reshape(1, -1)
    score = cosine_similarity(landmarks1, landmarks2)[0][0]
    return score
