import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tensorflow.keras.applications.resnet50 import preprocess_input

#load ResNet50 model trained on grayscale images (replicated to 3 channels)
best_model = load_model('resnet_sign_language_best_model.h5')

#load GPT-2 and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

#word-level GPT-2 autocomplete
def clean_with_gpt2(letter_buffer, max_tokens=5):
    raw_text = "".join(letter_buffer)
    words = raw_text.strip().split()
    if not words:
        return ""

    *prefix_words, current = words
    prompt = " ".join(prefix_words) + (" " if prefix_words else "") + current
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result_words = decoded.strip().split()

    if not result_words:
        return raw_text

    #replace current word with completed one
    if len(result_words) <= len(prefix_words):
        return raw_text

    completed_word = result_words[len(prefix_words)]
    return " ".join(prefix_words + [completed_word])

#class mapping
mapping = {
    0: 'A', 1: 'B', 2: 'K', 3: 'L', 4: 'M', 5: 'N', 6: 'O', 7: 'P', 8: 'Q', 9: 'R',
    10: 'S', 11: 'T', 12: 'C', 13: 'U', 14: 'V', 15: 'W', 16: 'X', 17: 'Y',
    18: 'D', 19: 'E', 20: 'F', 21: 'G', 22: 'H', 23: 'I'
}

#mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def get_hand_roi(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    roi, bbox = None, None
    if results.multi_hand_landmarks:
        h, w, _ = img.shape
        lm = results.multi_hand_landmarks[0]
        x_coords = [p.x for p in lm.landmark]
        y_coords = [p.y for p in lm.landmark]
        x_min = max(int(min(x_coords) * w) - 20, 0)
        y_min = max(int(min(y_coords) * h) - 20, 0)
        x_max = min(int(max(x_coords) * w) + 20, w)
        y_max = min(int(max(y_coords) * h) + 20, h)
        roi = img[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)
    return roi, bbox

#buffers and prediction vars
letter_buffer = []
consistent_prediction = ""
consistency_count = 0
CONSISTENCY_THRESHOLD = 10
gpt2_output = ""

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    roi, bbox = get_hand_roi(frame)
    prediction_text = ""

    if roi is not None:
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_roi, (224, 224))

        #since it's already 3 channels, no need to stack
        processed_img = preprocess_input(resized.astype("float32"))
        processed_img = np.expand_dims(processed_img, axis=0)

        pred = best_model.predict(processed_img)
        class_idx = int(np.argmax(pred, axis=1)[0])
        prediction_text = mapping.get(class_idx, "")

    #stability filter
    if prediction_text:
        if prediction_text == consistent_prediction:
            consistency_count += 1
        else:
            consistent_prediction = prediction_text
            consistency_count = 1

        if consistency_count >= CONSISTENCY_THRESHOLD:
            letter_buffer.append(consistent_prediction)
            consistency_count = 0
            consistent_prediction = ""
    else:
        consistent_prediction = ""
        consistency_count = 0

    #ui drawing
    if bbox:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    panel_height = 150
    panel = np.zeros((panel_height, frame.shape[1], 3), dtype="uint8")

    #style settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    line_thickness = 2
    text_color = (255, 255, 255)
    label_color = (100, 255, 200)
    highlight_color = (255, 255, 100)

    #draw background panel with border
    cv2.rectangle(panel, (10, 10), (frame.shape[1] - 10, panel_height - 10), (30, 30, 30), -1)
    cv2.rectangle(panel, (10, 10), (frame.shape[1] - 10, panel_height - 10), (80, 80, 80), 2)

    #display user-signed buffer
    cv2.putText(panel, "You Signed:", (30, 50), font, 0.8, label_color, 2)
    cv2.putText(panel, "".join(letter_buffer), (200, 50), font, font_scale, text_color, line_thickness)

    #display GPT-2 cleaned-up output
    cv2.putText(panel, "GPT-2 Suggestion:", (30, 110), font, 0.8, highlight_color, 2)
    cv2.putText(panel, gpt2_output, (330, 110), font, font_scale, text_color, line_thickness)

    #combine webcam frame with UI panel
    combined = np.vstack([frame, panel])


    cv2.imshow("ResNet50 Sign Language + GPT-2 Word Completion", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  #esc
        break
    elif key == ord('r'):  #reset
        letter_buffer = []
        consistent_prediction = ""
        consistency_count = 0
        gpt2_output = ""
    elif key == 13:  #ENTER = autocomplete last word
        if letter_buffer:
            gpt2_output = clean_with_gpt2(letter_buffer)
        else:
            gpt2_output = "[No input]"
    elif key == ord(' '):  #SPACEBAR = insert space
        letter_buffer.append(' ')
    elif key == 8:  # Backspace to delete last character
        if letter_buffer:
            letter_buffer.pop()
    elif key == ord('s'):  # 's' key to auto-complete the last word
        if letter_buffer:
            completed_text = clean_with_gpt2(letter_buffer)
            letter_buffer = list(completed_text)  # Replace buffer with completed word

cap.release()
cv2.destroyAllWindows()