import streamlit as st
import cv2
import time

from ultralytics import YOLO

from risk_score import calculate_risk
from emotion_detection import detect_emotion
from adaptive_signal import get_signal_time
from accident_prediction import detect_accident

# =========================
# PAGE TITLE
# =========================

st.title("AI Helmet Traffic System")

st.write("Live Helmet Detection Dashboard")

# =========================
# LOAD YOLO MODEL
# =========================

model = YOLO(
    r"C:\Users\ranja\runs\detect\train-2\weights\best.pt"
)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    st.error("Camera not working")

    st.stop()

# =========================
# STREAMLIT PLACEHOLDERS
# =========================

frame_placeholder = st.empty()

status_text = st.empty()

# =========================
# SETTINGS
# =========================

CONF_THRESHOLD = 0.50

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:

        st.write("Frame not captured")

        break

    # Flip webcam
    frame = cv2.flip(frame, 1)

    # =========================
    # YOLO DETECTION
    # =========================

    results = model(frame, verbose=False)

    helmet_detected = False
    no_helmet_detected = False

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # =========================
            # REVERSED LABEL FIX
            # =========================
            # YOUR MODEL:
            # 0 = no_helmet
            # 1 = helmet
            # =========================

            if cls == 1:

                helmet_detected = True

                # GREEN BOX
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Helmet {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            elif cls == 0:

                no_helmet_detected = True

                # RED BOX
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"No Helmet {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    # =========================
    # SIGNAL LOGIC
    # =========================

    if helmet_detected:

        signal = "GREEN"

        message = "Helmet Detected"

        color = (0, 255, 0)

    elif no_helmet_detected:

        signal = "RED"

        message = "No Helmet Detected"

        color = (0, 0, 255)

    else:

        signal = "YELLOW"

        message = "No Detection"

        color = (0, 255, 255)

    # =========================
    # RISK SCORE
    # =========================

    risk_score = calculate_risk(
        helmet=helmet_detected,
        triple_riding=False,
        mobile_usage=False,
        overspeed=False
    )

    # =========================
    # EMOTION
    # =========================

    emotion = "Unknown"

    if no_helmet_detected:

        emotion = detect_emotion(frame)

    # =========================
    # ADAPTIVE SIGNAL
    # =========================

    green_time = get_signal_time(
        risk_score
    )

    # =========================
    # ACCIDENT PREDICTION
    # =========================

    accident_risk = detect_accident(
        90,
        True
    )

    # =========================
    # DRAW SIGNAL
    # =========================

    cv2.circle(
        frame,
        (100, 100),
        40,
        color,
        -1
    )

    # =========================
    # DISPLAY TEXT
    # =========================

    cv2.putText(
        frame,
        f"Signal: {signal}",
        (40, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        message,
        (40, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Risk Score: {risk_score}",
        (40, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (40, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Signal Timer: {green_time}s",
        (40, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # =========================
    # ACCIDENT ALERT
    # =========================

    if accident_risk:

        cv2.putText(
            frame,
            "ACCIDENT RISK!",
            (40, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # =========================
    # CONVERT TO RGB
    # =========================

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # =========================
    # SHOW FRAME IN STREAMLIT
    # =========================

    frame_placeholder.image(
        frame,
        channels="RGB"
    )

    # =========================
    # STATUS PANEL
    # =========================

    status_text.markdown(f"""
    ### Live Status

    - Signal : {signal}
    - Message : {message}
    - Risk Score : {risk_score}
    - Emotion : {emotion}
    - Signal Timer : {green_time}s
    """)

cap.release()