import os
from collections import Counter, deque

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise ImportError(
        "mediapipe is not installed in the active environment. "
        "Install it with: pip install mediapipe"
    ) from exc

from tensorflow.keras.models import load_model

# ==========================
# Load trained model
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL_PATH = os.path.join(BASE_DIR, "hand_gesture_model.h5")
HAND_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

if not os.path.exists(GESTURE_MODEL_PATH):
    raise FileNotFoundError(
        f"Gesture model not found: {GESTURE_MODEL_PATH}"
    )

if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
    raise FileNotFoundError(
        "Hand Landmarker model not found: "
        f"{HAND_LANDMARKER_MODEL_PATH}. Download the .task model and place it "
        "next to this script."
    )

model = load_model(GESTURE_MODEL_PATH)

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

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

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

BaseOptions = mp_python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

hand_landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

hand_landmarker = HandLandmarker.create_from_options(hand_landmarker_options)


def draw_hand_landmarks(frame, landmarks):
    points = []
    height, width, _ = frame.shape

    for landmark in landmarks:
        x_coord = int(landmark.x * width)
        y_coord = int(landmark.y * height)
        points.append((x_coord, y_coord))
        cv2.circle(frame, (x_coord, y_coord), 4, (255, 0, 0), -1)

    for start_index, end_index in HAND_CONNECTIONS:
        if start_index < len(points) and end_index < len(points):
            cv2.line(frame, points[start_index], points[end_index], (0, 255, 0), 2)

    return points

# ==========================
# Webcam
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam at index 0.")

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = hand_landmarker.detect(mp_image)

        final_prediction = "No Hand"

        if results.hand_landmarks:

            hand_landmarks = results.hand_landmarks[0]
            points = draw_hand_landmarks(frame, hand_landmarks)

            # ======================
            # Bounding box
            # ======================

            x = [point[0] for point in points]
            y = [point[1] for point in points]

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

                img = img.astype(np.float32) / 255.0

                img = np.expand_dims(img, axis=0)

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
                    prediction_history.append(current_gesture)

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

finally:
    # ==========================
    # Cleanup
    # ==========================

    cap.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()