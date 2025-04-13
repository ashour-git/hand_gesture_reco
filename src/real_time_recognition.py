import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model('../models/gesture_model.h5')
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            hand_image = frame[y_min:y_max, x_min:x_max]
            if hand_image.size == 0:
                continue
            resized_hand = cv2.resize(hand_image, (64, 64))
            gray_hand = cv2.cvtColor(resized_hand, cv2.COLOR_BGR2GRAY)
            normalized_hand = gray_hand / 255.0
            input_data = normalized_hand.reshape(1, 64, 64, 1)
            prediction = model.predict(input_data)
            gesture = class_names[np.argmax(prediction)]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'Gesture: {gesture}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
