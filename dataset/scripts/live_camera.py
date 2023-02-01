import os.path

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import time
from PIL import Image

from dataset.scripts.utils import *

gesture_images_path = os.path.join('documentation', 'images', 'gestures')


def record_live(user, gestures_to_do=None):
    # Record gestures listed in gestures_to_do for use
    # Create a video for every gesture, it stops when 150 frames are recorded
    # If gestures_to_do is not provided record all gestures
    wait_time = 3  # time in seconds to wait between a gesture and the other
    frame_to_record = 200  # frame to record for every gesture

    # Load the dataset if already exists
    save_file = os.path.join('dataset', user + ".csv")
    if os.path.exists(save_file):
        df = pd.read_csv(save_file, index_col=0)
        gestures_recorded = df.value_counts("class")[df.value_counts("class") == frame_to_record].index.tolist()
    else:
        df = pd.DataFrame()
        gestures_recorded = []

    if gestures_to_do is None:
        gestures_to_do = [gest + "_right" for gest in labels] + \
                         [gest + "_left" for gest in labels]
        gestures_to_do = [gest for gest in gestures_to_do if gest not in gestures_recorded]

    # Create dataset/{user} folder if not exists
    user_folder = os.path.join('dataset', user)
    if not os.path.isdir(user_folder):
        os.mkdir(user_folder)

    # initialize mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # initialize webcam
    cap = cv2.VideoCapture(0)

    # Define the codec to save the video recorded
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    for gesture in gestures_to_do:
        # Create a video for every gesture
        gesture_video = os.path.join(user_folder, gesture + ".avi")
        out = cv2.VideoWriter(gesture_video, fourcc, fps, (frame_width, frame_height), )

        gest, hand = gesture.split('_')
        hand = 'DESTRA' if hand == 'right' else 'SINISTRA'
        gesture_image = cv2.imread(os.path.join(gesture_images_path, gest + '.png'), cv2.IMREAD_UNCHANGED)
        x_offset = gesture_image.shape[0]

        text_1 = f"Registra questo gesto con la mano {hand}"
        text_2 = "Premi la barra spaziatrice per registrare"

        title = f"Registra {gest} con la mano {hand}"

        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Show image of the gesture
            image = overlay_image(frame, gesture_image)

            show_text(image, text_1, x_offset, 50)
            show_text(image, text_2, x_offset, 90)

            cv2.imshow(title, image)
            key = cv2.waitKey(1)
            if key == ord(' '):
                break
            elif key == ord('q'):
                return

        # A timeout to let user prepare for next gesture
        start_time = time.time()
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            text = f"La registrazione partira' tra {wait_time - int(elapsed_time)} secondi..."

            _, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image = overlay_image(frame, gesture_image)

            show_text(image, text, x_offset, 50)
            cv2.imshow(title, image)

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
            out.write(frame)
            image = overlay_image(frame, gesture_image)

            result = hands.process(frame)

            # post process the result
            if result.multi_hand_landmarks:
                # Frame is recognized so save the landmarks
                frame_recorded += 1

                landmarks = []
                hand_lms = result.multi_hand_landmarks[0]

                for enu, lm in enumerate(hand_lms.landmark):
                    landmark_xyz = [lm.x, lm.y, lm.z]
                    landmarks.append(landmark_xyz)

                # normalize landmarks
                landmarks = normalize(landmarks)
                dictionary = create_dict(landmarks, gesture)

                df = pd.concat([df, pd.DataFrame([dictionary])], ignore_index=True)

                # draw landmarks on screen
                mp_draw.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)

            show_text(image, "Muovi leggermente la mano", x_offset, 50)

            # Show the final output
            cv2.imshow(title, image)
            if cv2.waitKey(1) == ord('q'):
                return

        # save the dataset
        df.to_csv(save_file)

        # save the video and destroy all active windows
        out.release()
        cv2.destroyAllWindows()

    # destroy the webcam
    cap.release()

    df.to_csv(save_file)


def overlay_image(img, img_overlay, x=0, y=0):
    """Overlay 'img_overlay' onto 'img' at (x, y)
    """

    image = Image.fromarray(img)
    overlay = Image.fromarray(img_overlay)

    image.paste(overlay, (x, y), overlay)

    return np.array(image)


def show_text(image, text, x, y):
    """Show 'text' onto 'img' at (x, y)
    """

    # Add some fading by showing the text in black translated
    cv2.putText(image, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
