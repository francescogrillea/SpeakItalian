import os.path

import cv2
import mediapipe as mp
import pandas as pd
import time

from dataset.scripts.utils import *


def record_live(user, gestures_to_do=None):
    # Record gestures listed in gestures_to_do for use
    # Create a video for every gesture, it stops when 150 frames are recorded
    # If gestures_to_do is not provided record all gestures
    wait_time = 3           # time in seconds to wait between a gesture and the other
    frame_to_record = 200   # frame to record for every gesture

    df = pd.DataFrame()

    if gestures_to_do is None:
        gestures_to_do = [gest + "_right" for gest in labels] + \
                         [gest + "_left" for gest in labels]

    # Create dataset/{user} folder if not exists
    user_folder = os.path.join('dataset', user)
    if not os.path.isdir(user_folder):
        os.mkdir(user_folder)

    # initialize mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    # Define the codec
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    for gesture in gestures_to_do:
        # Create video for every gesture
        gesture_video = os.path.join(user_folder, gesture + ".avi")
        out = cv2.VideoWriter(gesture_video, fourcc, fps, (frame_width, frame_height))

        # A timeout to let user prepare for next gesture
        start_time = time.time()
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            text = f"Recording {gesture} in {wait_time - int(elapsed_time)} seconds..."
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) == ord('q'):
                return

            if elapsed_time > wait_time:
                break

        frame_recorded = 0
        while frame_recorded < frame_to_record:
            # Read each frame from the webcam
            is_frame_read, frame = cap.read()

            if not is_frame_read:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            result = hands.process(frame)

            # post process the result
            if result.multi_hand_landmarks:
                # Frame is recognized so save it in the video and save the landmarks
                frame_recorded += 1
                out.write(frame)

                landmarks = []
                hand_lms = result.multi_hand_landmarks[0]

                for enu, lm in enumerate(hand_lms.landmark):
                    landmark_xyz = [lm.x, lm.y, lm.z]
                    landmarks.append(landmark_xyz)

                # normalize landmarks
                landmarks = normalize(landmarks)
                dictionary = create_dict(landmarks, gesture)

                df = pd.concat([df, pd.DataFrame([dictionary])], ignore_index=True)

                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Display gesture to record on screen
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # Show the final output
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) == ord('q'):
                return

        # release the webcam and destroy all active windows
        out.release()
        cv2.destroyAllWindows()

    cap.release()

    df.to_csv(os.path.join('dataset', user + ".csv"))
