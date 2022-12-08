import cv2
import mediapipe as mp
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
from utils import *


user = "francesco"
gesture = "thumbUp"

df = pd.DataFrame()

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
now = datetime.now() - timedelta(0, 10)


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)


    results = hands.process(frame)
    #print(results.multi_hand_landmarks)


    # Show results every x seconds
    sleep_time = 0.5
    if datetime.now() > now + timedelta(0, sleep_time):
        dictionary = {}

        if results.multi_hand_landmarks:
            landmarks = []
            handLms = results.multi_hand_landmarks[0]

            for enu, lm in enumerate(handLms.landmark):
                landmark_xyz = [lm.x, lm.y, lm.z]
                landmarks.append(landmark_xyz)

            landmarks = normalize(landmarks)
            dictionary = create_dict(landmarks)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            df = pd.concat([df, pd.DataFrame([dictionary])], ignore_index=True)
            print(landmarks)

        now = datetime.now()

    #exit condition
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()



print(df)
df.to_csv("../"+user+"_"+gesture+".csv")





