# This works if we take out the commands calling to setbase and strokedet for now. This is because there is an issue with webcams in that we are calling it twice-once throught the UI and the algorithm itself.
# Update! Make sure to comment out cv2.imshow commands in strokedet_allfun.py and then this will work! To increase the amount of time that the output is displayed, we can increase the number of seconds in the time.sleep() command
import streamlit as st
import pandas as pd
import PySimpleGUI as sg
import mediapipe as mp
import numpy as np
import time
from shapely.geometry import Polygon
import math
from scipy import stats
from strokedet_allfun import strokedet
from strokedet_allfun import setbase
from streamlit_webrtc import webrtc_streamer
import av
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.7)
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
def detect():
    result, r = strokedet()
    if result == 0:
        st.write('Stroke Unlikely')
        st.write('Percent Confidence:', r*100)
        time.sleep(5)
    else:
        st.write('Stroke Likely')
        st.write('Percent Confidence:', r*100)
        time.sleep(5)
class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        fm_result = face_mesh.process(frm)
        if fm_result.multi_face_landmarks:
            for face_landmarks in fm_result.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frm,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
        hands_result = hands.process(frm)
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frm,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        return av.VideoFrame.from_ndarray(frm, format = 'bgr24')
webrtc_streamer(key = "key", video_processor_factory=VideoProcessor)
st.title('Stroke Alert!')
with st.sidebar:
    st.title("Choose an option below")
    st.button("Set Baseline",  on_click = setbase)
    st.button("Run Detector!!", on_click = detect)