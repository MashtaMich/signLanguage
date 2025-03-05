import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp

# Load the trained model (expects color images with shape 224x224x3)
best_model = load_model('cnn_sign_language_best_model.h5')

# Mapping from class index to letter (adjust based on your classes)
mapping = {i: chr(65+i) for i in range(5)}  # e.g., 0 -> 'A', 1 -> 'B', etc.

# Initialize Mediapipe Hands for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

drawing_utils = mp.solutions.drawing_utils

# Function to extract hand ROI (as used in your image collection file)
def get_hand_roi(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    roi = None
    bbox = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = img.shape
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x_min = int(min(x_coords) * w)
        y_min = int(min(y_coords) * h)
        x_max = int(max(x_coords) * w)
        y_max = int(max(y_coords) * h)
        # Add some padding
        pad = 20
        x_min = max(x_min - pad, 0)
        y_min = max(y_min - pad, 0)
        x_max = min(x_max + pad, w)
        y_max = min(y_max + pad, h)
        roi = img[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)
    return roi, bbox

# Variables for consistency-based prediction
recognized_word = ""
consistent_prediction = ""
consistency_count = 0
CONSISTENCY_THRESHOLD = 5  # Consecutive frames required

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    # Get the hand ROI and its bounding box
    roi, bbox = get_hand_roi(frame)
    prediction_text = ""
    
    if roi is not None:
        # Resize the ROI to 224x224 in color (do NOT convert to grayscale)
        processed_img = cv2.resize(roi, (224, 224))
        processed_img = processed_img.astype("float32") / 255.0
        processed_img = np.expand_dims(processed_img, axis=0)  # Shape: (1, 224, 224, 3)
        
        # Predict with the model
        pred = best_model.predict(processed_img)
        class_idx = int(np.argmax(pred, axis=1)[0])
        prediction_text = mapping.get(class_idx, "")
    
    # Consistency check: update only after stable predictions
    if prediction_text:
        if prediction_text == consistent_prediction:
            consistency_count += 1
        else:
            consistent_prediction = prediction_text
            consistency_count = 1
        
        if consistency_count >= CONSISTENCY_THRESHOLD:
            recognized_word += consistent_prediction
            consistency_count = 0
            consistent_prediction = ""
    else:
        consistent_prediction = ""
        consistency_count = 0

    # Optionally, draw the bounding box around the detected hand
    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Create a blackboard to display the recognized word
    board = np.zeros((100, frame.shape[1], 3), dtype="uint8")
    cv2.putText(board, recognized_word, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Combine the live frame with the blackboard and display
    combined = np.vstack([frame, board])
    cv2.imshow("Real-Time Sign Recognition", combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('r'):  # Reset recognized word
        recognized_word = ""
        consistent_prediction = ""
        consistency_count = 0

cap.release()
cv2.destroyAllWindows()