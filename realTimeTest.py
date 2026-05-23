import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter

# ==========================
# Load trained model
# ==========================

model = load_model("hand_gesture_model.h5")

# IMPORTANT:
# Must match train_data.class_indices order

class_names = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down"
]

IMG_SIZE = 224

# ==========================
# Smoothing settings
# ==========================

# Keep a rolling window of the last N predictions
# for majority-vote smoothing
SMOOTHING_WINDOW = 10
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

# Only accept predictions above this confidence
CONFIDENCE_THRESHOLD = 0.80

# ==========================
# MediaPipe setup
# ==========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================
# Webcam
# ==========================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands_detector.process(rgb)

    final_prediction = "No Hand"

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # ======================
        # Bounding box
        # ======================

        x = [
            int(lm.x * w)
            for lm in hand_landmarks.landmark
        ]

        y = [
            int(lm.y * h)
            for lm in hand_landmarks.landmark
        ]

        padding = 20

        x_min = max(min(x) - padding, 0)
        x_max = min(max(x) + padding, w)

        y_min = max(min(y) - padding, 0)
        y_max = min(max(y) + padding, h)

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # ======================
        # Crop hand from RGB
        # ======================

        hand_img = rgb[
            y_min:y_max,
            x_min:x_max
        ]

        if hand_img.size != 0:

            # ======================
            # Preprocess
            # ======================

            img = cv2.resize(
                hand_img,
                (IMG_SIZE, IMG_SIZE)
            )

            img = img.astype(
                np.float32
            ) / 255.0

            img = np.expand_dims(
                img,
                axis=0
            )

            # ======================
            # Prediction (fast path)
            # ======================

            pred = model.predict_on_batch(img)

            confidence = float(np.max(pred))

            class_index = int(np.argmax(pred))

            # ======================
            # Confidence threshold
            # ======================

            if confidence >= CONFIDENCE_THRESHOLD:

                current_gesture = class_names[class_index]

                # Append to smoothing window
                prediction_history.append(
                    current_gesture
                )

                # ======================
                # Gesture smoothing
                # (majority vote over window)
                # ======================

                smoothed_prediction = Counter(
                    prediction_history
                ).most_common(1)[0][0]

                final_prediction = (
                    f"{smoothed_prediction} "
                    f"{confidence * 100:.1f}%"
                )

            else:
                # Below confidence threshold —
                # show that a hand is detected
                # but gesture is uncertain
                final_prediction = (
                    f"Uncertain "
                    f"{confidence * 100:.1f}%"
                )

    else:
        # No hand detected — clear history
        # so stale predictions don't persist
        prediction_history.clear()

    # ==========================
    # Display text
    # ==========================

    cv2.putText(
        frame,
        final_prediction,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Hand Gesture Recognition",
        frame
    )

    # Press Q to quit

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# Cleanup
# ==========================

cap.release()
cv2.destroyAllWindows()