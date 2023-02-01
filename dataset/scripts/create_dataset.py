import os

import cv2
import mediapipe as mp
import time
import pandas as pd
from dataset.scripts.utils import *


def create_dataset(*users, flip_horizontally=False, save_file=None):
    # Given a list of users automatically create the dataset with all the gestures for every user
    # It searches the gestures in dataset/{user}

    for user in users:

        if save_file is None:
            save_file = os.path.join('dataset', f'{user}.csv')

        df = pd.DataFrame()

        for root, _, files in os.walk(os.path.join('dataset', user)):
            for file in files:
                if file.endswith('.avi') and not file.startswith('all_gestures'):
                    result = process_video(os.path.join(root, file), horizontal_flip=flip_horizontally)
                    df = pd.concat([df, result], ignore_index=True)

        df.to_csv(save_file)


def process_video(filename, save_df=False, horizontal_flip=False):
    print(f'processing {filename}...')

    df = pd.DataFrame()
    label = os.path.split(filename)[-1].removesuffix('.avi')

    cap = cv2.VideoCapture(filename)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1,
                          min_detection_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    while cap.isOpened():
        ret, frame = cap.read()
        if horizontal_flip:
            frame = cv2.flip(frame, 1)

        if not ret:
            break

        results = hands.process(frame)

        if results.multi_hand_landmarks:
            landmarks = []
            handLms = results.multi_hand_landmarks[0]

            for enu, lm in enumerate(handLms.landmark):
                landmark_xyz = [lm.x, lm.y, lm.z]
                landmarks.append(landmark_xyz)

            # normalize landmarks
            landmarks = normalize(landmarks)
            dictionary = create_dict(landmarks, label)

            df = pd.concat([df, pd.DataFrame([dictionary])], ignore_index=True)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            # Show the final output
            cv2.imshow(label, frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

    # print(df)
    if save_df:
        df.to_csv(filename[:-4]+".csv")

    return df
