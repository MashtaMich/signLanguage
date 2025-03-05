import cv2
import os
import random
import mediapipe as mp

# Desired image size for saving
image_x, image_y = 224, 224

def init_create_folder_database():
    # Create the "dataset" folder if it does not exist
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

# Initialize Mediapipe Hands for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def get_hand_roi(img):
    """
    Uses Mediapipe to detect a hand in the frame and extracts the ROI.
    Returns the ROI and the bounding box.
    """
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
        # Add some padding to the bounding box
        pad = 20
        x_min = max(x_min - pad, 0)
        y_min = max(y_min - pad, 0)
        x_max = min(x_max + pad, w)
        y_max = min(y_max + pad, h)
        roi = img[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)
    return roi, bbox

def store_images(g_id):
    total_pics = 200
    cam = cv2.VideoCapture(1)
    # Fallback to camera 0 if camera 1 is not available
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)
    # Create folder to save images for this gesture under "dataset"
    folder_path = os.path.join("dataset", str(g_id))
    create_folder(folder_path)
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        roi, bbox = get_hand_roi(img)
        # Only save images when capturing is enabled and after a few frames for stabilization
        if roi is not None and frames > 10 and flag_start_capturing:
            # Convert ROI to grayscale and resize to desired dimensions
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            save_img = cv2.resize(gray_roi, (image_x, image_y))
            # Randomly flip the image to add variation
            if random.randint(0, 10) % 2 == 0:
                save_img = cv2.flip(save_img, 1)
            pic_no += 1
            cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255), 2)
            cv2.imwrite(os.path.join(folder_path, f"{pic_no}.jpg"), save_img)

        if bbox is not None:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(img, f"Count: {pic_no}", (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255), 2)
        cv2.imshow("Capturing gesture", img)
        if roi is not None:
            cv2.imshow("Hand ROI", roi)
        keypress = cv2.waitKey(1) & 0xFF
        # Press 'c' to toggle capturing on/off
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            frames = 0  # Reset frame counter when toggling capture mode
        if flag_start_capturing:
            frames += 1
        # Press ESC to exit or break if enough images are captured
        if keypress == 27 or pic_no >= total_pics:
            break

    cam.release()
    cv2.destroyAllWindows()

# Initialize folders and database
init_create_folder_database()

g_id = input("Enter gesture no.: ")

store_images(g_id)