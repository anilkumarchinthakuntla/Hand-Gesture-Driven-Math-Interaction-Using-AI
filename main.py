import cv2
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")
st.title("✋ Handwritten Math Solver")

col1, col2 = st.columns([3, 2])

with col1:
    run = st.checkbox("Run Camera")
    frame_placeholder = st.image([])

with col2:
    st.subheader("Answer")
    answer_box = st.empty()

# ---------------- Gemini ----------------
genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- Hand Detector ----------------
detector = HandDetector(maxHands=1, detectionCon=0.7)

# ---------------- Helper Functions ----------------
def getHandInfo(img):
    result = detector.findHands(img, draw=False)
    hands = result[0] if isinstance(result, tuple) else result
    if hands:
        hand = hands[0]
        return detector.fingersUp(hand), hand["lmList"]
    return None, None


def draw(info, prev_pos, canvas):
    fingers, lmList = info
    curr_pos = None

    if fingers == [0, 1, 0, 0, 0]:
        curr_pos = lmList[8][:2]
        if prev_pos is not None:
            cv2.line(canvas, prev_pos, curr_pos, (255, 0, 255), 10)

    elif fingers == [1, 0, 0, 0, 0]:
        canvas[:] = 0

    return curr_pos, canvas


def sendToAI(canvas):
    pil_img = Image.fromarray(canvas)
    response = model.generate_content(
        ["Solve this handwritten math problem", pil_img]
    )
    return response.text


# ---------------- Session State ----------------
if "cap" not in st.session_state:
    st.session_state.cap = None
if "canvas" not in st.session_state:
    st.session_state.canvas = None
if "prev_pos" not in st.session_state:
    st.session_state.prev_pos = None
if "ai_called" not in st.session_state:
    st.session_state.ai_called = False

# ---------------- Main Logic ----------------
if run:
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(3, 640)
        st.session_state.cap.set(4, 480)

    success, img = st.session_state.cap.read()
    if success:
        img = cv2.flip(img, 1)

        if st.session_state.canvas is None:
            st.session_state.canvas = np.zeros_like(img)

        fingers, lmList = getHandInfo(img)

        if fingers is not None:
            st.session_state.prev_pos, st.session_state.canvas = draw(
                (fingers, lmList),
                st.session_state.prev_pos,
                st.session_state.canvas
            )

            if fingers == [1, 1, 1, 1, 0] and not st.session_state.ai_called:
                answer_box.text(sendToAI(st.session_state.canvas))
                st.session_state.ai_called = True

            if fingers != [1, 1, 1, 1, 0]:
                st.session_state.ai_called = False

        combined = cv2.addWeighted(
            img, 0.7, st.session_state.canvas, 0.3, 0
        )
        frame_placeholder.image(combined, channels="BGR")

else:
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
