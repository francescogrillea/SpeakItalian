import cv2
import mediapipe as mp
import time
from utils import *

def frame_acquisition(filename):
    cap = cv2.VideoCapture(filename)
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    results = hands.process(frame)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        landmarks = []

        for handLms in results.multi_hand_landmarks:
            for enu, lm in enumerate(handLms.landmark):
                landmark_xyz = [lm.x, lm.y, lm.z]
                print([f"{c:0.3f}" for c in landmark_xyz], end='\t')
                print(landmarks_name[enu])

                landmarks.append(landmark_xyz)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        print("\nNORMALIZED")
        for enu, lm in enumerate(normalize(landmarks)):
            print([f"{c:0.3f}" for c in lm], end='\t')
            print(landmarks_name[enu])
        print()

    cv2.imshow("Image", frame)
    if cv2.waitKey(0) == ord('q'):
        # release the webcam and destroy all active windows
        cap.release()
        cv2.destroyAllWindows()
