import cv2
import mediapipe as mp
from datetime import datetime
from datetime import timedelta
import pandas as pd
from utils import *
import os

# DA TESTARE
def create_dataset(root_filepath):
    _, user = os.path.split(root_filepath)
    df = pd.DataFrame()

    videos = [file for file in os.listdir(root_filepath) if file.endswith('mp4')]
    for video in videos:
        df = pd.concat([df, process_stream(video)], ignore_index=True)

    df.to_csv(f"dataset/{user}.csv")

# DA TESTARE
def process_stream(filename, frame_rate=5, save_df=False):
    # Given in input the path of a video it extracts one frame every {frame_rate} frames
    # and associate to every frame the gesture contained in the name of the video
    # Video should be named like
    #   Daniele_thumbsUp.mp4

    dir_path, file = os.path.split(filename)
    user, gesture = file.removesuffix().split('_')
    path = os.path.join(dir_path, gesture)
    if not os.path.exists(path):
        os.mkdir(path)

    df = pd.DataFrame()

    cap = cv2.VideoCapture(filename)

    frame_counter = 0
    current_frame = frame_rate // 2

    while cap.isOpened():
        is_open, frame = cap.read()
        landmarks = get_landmarks(frame)

        if not is_open:
            break

        # Show results every {frame_rate} frames
        if current_frame > frame_rate and landmarks:
            current_frame = 0

            frame_file = path + f"{frame_counter}.jpg"
            cv2.imwrite(frame_file, frame)

            dictionary = create_dict(landmarks)
            frame_counter += 1

            df = pd.concat([df, pd.DataFrame([dictionary])], ignore_index=True)

        current_frame += 1

    # release the webcam and destroy all active windows
    cap.release()

    # print(df)
    if save_df:
        df.to_csv(filename[:-3] + ".csv")
    return df


def get_landmarks(frame, show_image=False):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    results = hands.process(frame)
    if not results.multi_hand_landmarks:
        return None

    landmarks = []
    for enu, lm in enumerate(results.multi_hand_landmarks[0].landmark):
        landmark_xyz = [lm.x, lm.y, lm.z]
        landmarks.append(landmark_xyz)

    if show_image:
        mp_draw = mp.solutions.drawing_utils
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()

    return normalize(landmarks)
