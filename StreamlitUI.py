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
from strokedet_allfun import strokedetII
from strokedet_allfun import setbase
from streamlit_webrtc import webrtc_streamer
import av
import cv2

class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        # frm = np.flip(frm,axis=1)
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
        return av.VideoFrame.from_ndarray(np.flip(frm,axis=1), format = 'bgr24')

def detect():
    r = strokedet()
    
    if r < 0.5:
        st.markdown("""<style>.big-font {font-size:50px !important;}</style>""", unsafe_allow_html=True)
        st.markdown('<p class="big-font">Raise your hands</p>', unsafe_allow_html=True)
        result = strokedetII(r)
    else:
        result = 0

    if result == 0:
        st.markdown("""<style>.big-font {font-size:50px !important;}</style>""", unsafe_allow_html=True)
        st.markdown('<p class="big-font">Stroke Unlikely !!</p>', unsafe_allow_html=True)
        st.write('Percent Confidence:', r*100)
        time.sleep(5)
    else:
        st.markdown("""<style>.big-font {font-size:300px !important;}</style>""", unsafe_allow_html=True)
        st.markdown('<p class="big-font">Stroke Likely !!</p>', unsafe_allow_html=True)
        st.write('Percent Confidence:', r*100)
        time.sleep(5)

# ==============================================

# tab feature source: https://github.com/streamlit/streamlit/issues/233#issuecomment-821873409

st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
    unsafe_allow_html=True,
)

query_params = st.experimental_get_query_params()
tabs = ["Home", "Directions", "Information"]
if "tab" in query_params:
    active_tab = query_params["tab"][0]
else:
    active_tab = "Home"

if active_tab not in tabs:
    st.experimental_set_query_params(tab="Home")
    active_tab = "Home"

li_items = "".join(
    f"""
    <li class="nav-item">
        <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}" target="_self">{t}</a>
    </li>
    """
    for t in tabs
)
tabs_html = f"""
    <ul class="nav nav-tabs">
    {li_items}
    </ul>
"""

st.markdown(tabs_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if active_tab == "Home":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.7)
    hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    st.write('Welcome to Stroke Alert! This application uses features from your face like your eyes, nose, and mouth to determine if you are having a stroke. It will also display how confident we are with our prediction.')
    st.write('To use this app, first click "Start" to view the real time video with the face tesselation, that is extracting facial features. If this is your first time using the app, click "Set Baseline" so the algorithm can record baseline data (30 seconds). If you have a stored baseline and you are ready to run the detector, click "Run Detector!!')
    st.write('To reset the app, click the "Stop" button below the video player')
    st.write('Before using the app, please read the instructions for use carefully! :https://docs.google.com/document/d/1B8YqBl4R1NvvypYeVmfYzLNlG5ipcKHK3ztidYRZ3C8/edit')

    webrtc_streamer(key = "key", video_processor_factory=VideoProcessor)
    with st.sidebar:
        st.title('Welcome to Stroke Alert!')
        st.title("Choose an option below")
        st.button("Set Baseline",  on_click = setbase)
        st.button("Run Detector!!", on_click = detect)
        
elif active_tab == "Directions":
    st.write("Directions")

elif active_tab == "Information":
    st.write("Information")
    st.markdown()

else:
    st.error("Something has gone wrong.")

# ==============================================

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_mesh = mp.solutions.face_mesh
# mp_hands = mp.solutions.hands

# face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.7)
# hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# st.write('Welcome to Stroke Alert! This application uses features from your face like your eyes, nose, and mouth to determine if you are having a stroke. It will also display how confident we are with our prediction.')
# st.write('To use this app, first click "Start" to view the real time video with the face tesselation, that is extracting facial features. If this is your first time using the app, click "Set Baseline" so the algorithm can record baseline data (30 seconds). If you have a stored baseline and you are ready to run the detector, click "Run Detector!!')
# st.write('To reset the app, click the "Stop" button below the video player')
# st.write('Before using the app, please read the instructions for use carefully! :https://docs.google.com/document/d/1B8YqBl4R1NvvypYeVmfYzLNlG5ipcKHK3ztidYRZ3C8/edit')




# webrtc_streamer(key = "key", video_processor_factory=VideoProcessor)
# with st.sidebar:
#     st.title('Welcome to Stroke Alert!')
#     st.title("Choose an option below")
#     st.button("Set Baseline",  on_click = setbase)
#     st.button("Run Detector!!", on_click = detect)
#     st.button("Directions", )