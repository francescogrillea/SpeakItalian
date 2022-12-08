import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,1)


    results = hands.process(frame)
    #print(results.multi_hand_landmarks)


    if results.multi_hand_landmarks:
        landmarks = []

        for handLms in results.multi_hand_landmarks:
            for lm in handLms.landmark:
                #print(id,lm)
                landmarks.append([lm.x, lm.y, lm.z])


            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        print(landmarks)


    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
