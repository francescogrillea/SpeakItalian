import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import pickle as pk
from dataset.scripts.utils import normalize, create_dict, show_text


BATCH_SIZE = 10
FEATURE_SIZE = 9

pipe = pk.load(open('model/pipeline.sav', 'rb'))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


# initialize webcam
cap = cv2.VideoCapture(0)

batch = pd.DataFrame()
label = ""

while True:

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)


    results = hands.process(frame)
    #print(results.multi_hand_landmarks)


    if results.multi_hand_landmarks:
        landmarks = []
        hand_lms = results.multi_hand_landmarks[0]

        for enu, lm in enumerate(hand_lms.landmark):
            landmarks.append([lm.x, lm.y, lm.z])

        # normalize landmarks
        landmarks = normalize(landmarks)

        dictionary = create_dict(landmarks, None)
        new_data = pd.DataFrame([dictionary]).drop(columns=["class"])

        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        batch = pd.concat([batch, pd.DataFrame(new_data)], ignore_index=True)

        if batch.shape[0] >= BATCH_SIZE:
            output = pipe.predict(batch)
            u, c = np.unique(output, return_counts=True)
            label = u[c.argmax()]
            print(label)

            batch = pd.DataFrame()

    show_text(frame, label)

    cv2.imshow("SpeakItalian", frame)
    try:
        if cv2.waitKey(1) == ord('q'):
            break
    except Exception:
        break

cap.release()