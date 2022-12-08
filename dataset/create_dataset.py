import cv2
import mediapipe as mp
import time
from datetime import datetime
from datetime import timedelta
from landmarks import landmarks_name


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
    sleep_time = 3
    if datetime.now() > now + timedelta(0, sleep_time):

        if results.multi_hand_landmarks:
            landmarks = []

            for handLms in results.multi_hand_landmarks:
                for enu, lm in enumerate(handLms.landmark):
                    landmark_xyz = [lm.x, lm.y, lm.z]
                    print([f"{c:0.3f}" for c in landmark_xyz], end='\t')
                    print(landmarks_name[enu])


                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            print(landmarks)

        now = datetime.now()

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
