import cv2
import mediapipe as mp
import numpy as np
import time
from shapely.geometry import Polygon
import math
import os, os.path
from scipy import stats
import glob
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_drawing_spec = mp_drawing.DrawingSpec(color=[244,244,244],thickness=1)
cap = cv2.VideoCapture(0)

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.7)
hands = mp_hands.Hands(max_num_hands=10,model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fm_result = face_mesh.process(image)
    hands_result = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if fm_result.multi_face_landmarks:
        for face_landmarks in fm_result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=face_drawing_spec)
            # mp_drawing.draw_landmarks(
            #     image=image, 
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None, 
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Landmarks',cv2.flip(image, 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
