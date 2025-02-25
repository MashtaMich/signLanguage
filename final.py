import cv2, sqlite3, pyttsx3
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model
from threading import Thread
import mediapipe as mp

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Load the trained model (update filename as needed)
model = load_model('sign_language_model_regularized-2.h5')

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def say_text(text):
    engine.say(text)
    engine.runAndWait()

def get_image_size():
    # Our model expects images of size 50x50.
    return (50, 50)

image_x, image_y = get_image_size()

def keras_process_image(img):
    # Resize and reshape the ROI to match model input shape: (1, 50, 50, 1)
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
    # Query the database for the gesture name corresponding to pred_class
    conn = sqlite3.connect("gesture2_db.db")
    query = "SELECT g_name FROM gesture WHERE g_id=?"
    cursor = conn.execute(query, (pred_class,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return ""

def get_pred_from_roi(roi):
    # Ensure the ROI is in grayscale
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi
    h, w = gray_roi.shape
    # Adjust ROI to be square by padding if necessary
    if w > h:
        diff = w - h
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        gray_roi = cv2.copyMakeBorder(gray_roi, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
    elif h > w:
        diff = h - w
        pad_left = diff // 2
        pad_right = diff - pad_left
        gray_roi = cv2.copyMakeBorder(gray_roi, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    pred_probab, pred_class = keras_predict(model, gray_roi)
    text = ""
    if pred_probab * 100 > 70:  # Use 70% as the confidence threshold
        text = get_pred_text_from_db(pred_class)
    return text

def get_hand_roi(img):
    """
    Uses MediaPipe to detect a hand in the frame, computes a padded bounding box,
    and extracts the ROI.
    """
    # Flip the image for a mirror view and convert to RGB
    img_flipped = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    roi = None
    bbox = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = img_flipped.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        # Add padding to the bounding box
        padding = 20
        x_min = max(x_min - padding, 0)
        y_min = max(y_min - padding, 0)
        x_max = min(x_max + padding, w)
        y_max = min(y_max + padding, h)
        roi = img_flipped[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)
        # Draw hand landmarks for visualization
        mp_draw.draw_landmarks(img_flipped, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return img_flipped, roi, bbox

def text_mode(cam):
    text = ""
    word = ""
    count_same_frame = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        img, roi, bbox = get_hand_roi(frame)
        old_text = text
        if roi is not None:
            # Only consider ROI if its area is large enough
            if roi.shape[0] * roi.shape[1] > 10000:
                text = get_pred_from_roi(roi)
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0
            else:
                if word != "":
                    Thread(target=say_text, args=(word,)).start()
                text = ""
                word = ""
        else:
            if word != "":
                Thread(target=say_text, args=(word,)).start()
            text = ""
            word = ""
        # Create a blackboard to display prediction and accumulated text
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Predicted text: " + text, (30, 100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word, (30, 240),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        if roi is not None:
            cv2.imshow("Hand ROI", roi)
        keypress = cv2.waitKey(1)
        if keypress == ord('q') or keypress == ord('c'):
            break
    if keypress == ord('c'):
        return 2
    else:
        return 0

def recognize():
    cam = cv2.VideoCapture(1)
    # Fallback to camera 0 if camera 1 is unavailable
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)
    text_mode(cam)

# Warm up the model with a dummy image before starting recognition
keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
recognize()