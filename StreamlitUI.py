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
    print(r)
    opr = 100*r
    
    if r < 0.5:
        st.markdown("""<style>.big-font {font-size:50px !important;}</style>""", unsafe_allow_html=True)
        st.markdown('<p class="big-font">Raise your hands</p>', unsafe_allow_html=True)
        ruling = strokedetII(r)
        time.sleep(1)
    else:
        ruling = 0

    if ruling == 0:
        st.markdown("""<style>.big-font {font-size:0px !important;}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>.stroke-unlikely {font-size:50px !important;}</style>""", unsafe_allow_html=True)
        st.markdown('<p class="stroke-unlikely">Stroke Unlikely !!</p>', unsafe_allow_html=True)
        st.write("Percent Confidence: %.2f" % opr)
        time.sleep(5)
    else:
        st.markdown("""<style>.big-font {font-size:0px !important;}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>.stroke-likely {font-size:50px !important;}</style>""", unsafe_allow_html=True)
        st.markdown('<p class="stroke-likely">Stroke Likely !!</p>', unsafe_allow_html=True)
        st.write("Percent Confidence: %.2f" % opr)
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
    
    webrtc_streamer(key = "key", video_processor_factory=VideoProcessor)
    with st.sidebar:
        st.title('Welcome to Stroke Alert!')
        st.title("Choose an option below")
        st.button("Set Baseline",  on_click = setbase)
        st.button("Run Detector!!", on_click = detect)
        
elif active_tab == "Directions":
    st.write('To reset the app, click the "Stop" button below the video player')
    st.markdown("## Directions for Use:")
    st.markdown("1. First press the start button to run the video player.")
    st.markdown("2. Upload a baseline if this is the first time you are using the application by pressing the ‘Set Baseline’ button.")
    st.markdown("3. Allow the algorithm to film for 30 seconds. Maintain proper lighting and eye contact with the camera to allow for best results.")
    st.markdown("4. After setting a baseline, the algorithm is ready to be used.")
    st.markdown("5. If you believe you may be having a stroke, press the ‘Run Detector’ button.")
    st.markdown("6. If the algorithm says “Stroke Unlikely”, continue to monitor symptoms.")
    st.markdown("7. If the algorithm says “Stroke Likely”, call EMS.")
elif active_tab == "Information":
    st.markdown("# Instructions for Use Stroke Alert- Stroke Detection Algorithm")
    st.markdown("## Device Name:")
    st.markdown("The device's brand name is Stroke Alert.")
    st.markdown("## Action Mechanism:")
    st.markdown("The purpose of this system is to analyze an individual’s facial characteristics to determine their likelihood of suffering a stroke at that moment. The system is intended to increase the user’s confidence in their symptoms being indicative of a potential stroke; this will be communicated to the user through a display on the device screen that contains the individual’s current calculated likelihood of stroke – displayed as a confidence interval.")
    st.markdown("## Indications:")
    st.markdown("The Stroke Alert is intended to encourage the user to seek medical attention when symptoms of stroke are recognized by the system. This is indicated for users who are at high risk for stroke. An individual’s risk for stroke can increase with genetic predisposition, a personal history of stroke, a family history of stroke, metabolic syndrome (high blood sugar, blood pressure, cholesterol, etc.), or other lifestyle factors that could be discussed with a general practitioner. ")
    st.markdown("## Contraindications:")
    st.markdown("This product should not be used as a primary diagnostic tool. If you are having a stroke, call EMS.")
    st.markdown("## Adverse Reactions:")
    st.markdown("No adverse reactions have been issued")
    st.markdown("## Warnings:")
    st.markdown("1. This device is not intended to be a stand alone diagnostic tool. If you are having a stroke, call EMS.")
    st.markdown("2. This device is not to be used in replacement of a physician.")
    st.markdown("## Precaution:")
    st.markdown("1. This device is not intended to be a stand alone diagnostic tool. If you are having a stroke, call EMS.")
    st.markdown("2. Clinicians and patients must follow the intended use of the device.")
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