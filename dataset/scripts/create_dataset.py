import cv2
import mediapipe as mp
from datetime import datetime
from datetime import timedelta
import pandas as pd
from utils import *


def create_dataset(root_filepath):
    df = pd.DataFrame()

    # TODO - browse recursively each video in folders
    #df = pd.concat([df, record_stream(filename, frame_rate)], ignore_index=True)

    df.to_csv("dataset.csv")



def record_stream(filename, frame_rate, save_df=False):

    df = pd.DataFrame()

    cap = cv2.VideoCapture(filename)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils


    while(cap.isOpened()):
        ret, frame = cap.read()
        results = hands.process(frame)

        if not ret:
            break


        # Show results every x seconds
        # TODO - use frame_rate
        sleep_time = 0.5
        if datetime.now() > now + timedelta(0, sleep_time):
            dictionary = {}

            if results.multi_hand_landmarks:
                landmarks = []
                handLms = results.multi_hand_landmarks[0]

                for enu, lm in enumerate(handLms.landmark):
                    landmark_xyz = [lm.x, lm.y, lm.z]
                    landmarks.append(landmark_xyz)

                # normalize landmarks
                landmarks = normalize(landmarks)
                dictionary = create_dict(landmarks)
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                df = pd.concat([df, pd.DataFrame([dictionary])], ignore_index=True)
                print(landmarks)

            now = datetime.now()

        #exit condition
        cv2.imshow("Image", frame)

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

    print(df)
    if save_df:
        df.to_csv(filename[:-3]+".csv")
    return df





# TODO - add execution
#create_dataset(root_filepath)



