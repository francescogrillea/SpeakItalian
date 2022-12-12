import os

import cv2
import mediapipe as mp
import time
import pandas as pd
from utils import *


def create_dataset(root_filepath):
    df = pd.DataFrame()

    for root, subdirs, files in os.walk(root_filepath):
        for file in files:
            result = record_stream(os.path.join(root, file))
            df = pd.concat([df, result], ignore_index=True)

    df.to_csv("dataset.csv")



def record_stream(filename, frame_rate=5, save_df=False):

    df = pd.DataFrame()
    label = filename.split("_")[-1].split(".")[0]

    cap = cv2.VideoCapture(filename)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    prev = 0

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        time_elapsed = time.time() - prev
        results = hands.process(frame)

        if time_elapsed > 1./frame_rate:

            if results.multi_hand_landmarks:
                landmarks = []
                handLms = results.multi_hand_landmarks[0]

                for enu, lm in enumerate(handLms.landmark):
                    landmark_xyz = [lm.x, lm.y, lm.z]
                    landmarks.append(landmark_xyz)

                # normalize landmarks
                landmarks = normalize(landmarks)
                dictionary = create_dict(landmarks, label)
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                df = pd.concat([df, pd.DataFrame([dictionary])], ignore_index=True)
                print(landmarks)

            prev = time.time()

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

    print(df)
    if save_df:
        df.to_csv(filename[:-3]+".csv")
    return df





create_dataset("../media/")



