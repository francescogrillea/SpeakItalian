import cv2
import mediapipe as mp
from datetime import datetime, timedelta
import pandas as pd
from utils import *
import os


user = "daniele"
gesture = "thumbUp"

path = f'../{user}/'
image_path = path + 'images/'
print(os.listdir())

if not os.path.exists(path):
    os.mkdir(path)

if not os.path.exists(image_path):
    print(image_path)
    os.mkdir(image_path)


df = pd.DataFrame()

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

current_frame = 5
# Show results every x frame
sleep_frame = 10

frame_counter = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    results = hands.process(frame)

    dictionary = {}

    if results.multi_hand_landmarks:
        landmarks = []
        handLms = results.multi_hand_landmarks[0]

        for enu, lm in enumerate(handLms.landmark):
            landmark_xyz = [lm.x, lm.y, lm.z]
            landmarks.append(landmark_xyz)

        landmarks = normalize(landmarks)
        dictionary = create_dict(landmarks)

        if current_frame > sleep_frame:
            current_frame = 0

            gesture_path = image_path + gesture + "/"
            if not os.path.exists(gesture_path):
                os.mkdir(gesture_path)

            filename = gesture_path + f"{frame_counter}.jpg"
            print(filename)
            cv2.imwrite(filename, frame)

            frame_counter += 1

            df = pd.concat([df, pd.DataFrame([dictionary])], ignore_index=True)
            print(landmarks)

        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    current_frame += 1
    print(current_frame)

    # exit condition
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()


# print(df)

df.to_csv(path + gesture + ".csv")





