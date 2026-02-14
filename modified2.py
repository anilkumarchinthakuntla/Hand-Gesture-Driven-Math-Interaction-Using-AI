import cv2
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai
import os
import time

# 1. STOP TERMINAL BLINKING/WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")
st.title("✋ Handwritten Math Solver")

col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox("Run Camera")
    # Using .empty() prevents the "flicker" during reruns
    frame_placeholder = st.empty() 

with col2:
    st.subheader("Answer")
    answer_box = st.empty()

# ---------------- Gemini ----------------
# NOTE: Ensure your API Key is valid and has quota
genai.configure(api_key="AIzaSyCavSU_qjt2t2hDfBuW5sgJfwsAZnZVyVU")
model = genai.GenerativeModel('gemini-3-flash-preview')

# ---------------- Session State ----------------
if "canvas" not in st.session_state:
    st.session_state.canvas = None
if "prev_pos" not in st.session_state:
    st.session_state.prev_pos = None

# Initialize Detector inside session state to prevent it from resetting
if "detector" not in st.session_state:
    st.session_state.detector = HandDetector(maxHands=1, detectionCon=0.7)

# ---------------- Main Logic ----------------
if run:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # THIGH LOOP: This keeps video smooth without refreshing the whole page
    while run:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if st.session_state.canvas is None:
            st.session_state.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Hand Detection (using session state detector)
        hands, frame = st.session_state.detector.findHands(frame, draw=True)
        
        if hands:
            hand = hands[0]
            fingers = st.session_state.detector.fingersUp(hand)
            curr = hand["lmList"][8][:2] # Index tip

            # DRAW: Index Up [0, 1, 0, 0, 0]
            if fingers == [0, 1, 0, 0, 0]:
                if st.session_state.prev_pos is not None:
                    cv2.line(st.session_state.canvas, st.session_state.prev_pos, curr, (255, 0, 255), 10)
                st.session_state.prev_pos = curr
            
            # CLEAR: Thumb Up [1, 0, 0, 0, 0]
            elif fingers == [1, 0, 0, 0, 0]:
                st.session_state.canvas = np.zeros_like(st.session_state.canvas)
                st.session_state.prev_pos = None

            # SOLVE: Open Palm [1, 1, 1, 1, 1]
            elif fingers == [1, 1, 1, 1, 1]:
                answer_box.info("Analysing...")
                try:
                    img_rgb = cv2.cvtColor(st.session_state.canvas, cv2.COLOR_BGR2RGB)
                    response = model.generate_content(["Solve the math in this image:", Image.fromarray(img_rgb)])
                    answer_box.success(f"Result: {response.text}")
                    time.sleep(2) # Prevent spamming API (resolves 429 error)
                except Exception as e:
                    answer_box.error(f"API Error: {e}")
            else:
                st.session_state.prev_pos = None
        else:
            st.session_state.prev_pos = None

        # Display the video frame in the stable placeholder
        combined = cv2.addWeighted(frame, 0.7, st.session_state.canvas, 0.4, 0)
        frame_placeholder.image(combined, channels="BGR")
        
        # Small sleep to stabilize FPS
        time.sleep(0.01)

    cap.release()
else:
    st.info("Check 'Run Camera' to start.")