import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volbar = 400
volper = 0

volMin, volMax, _ = volume.GetVolumeRange()


GESTURE_STATE_NONE = 0
GESTURE_STATE_VOLUME = 1
GESTURE_STATE_MUTE = 2
GESTURE_STATE_NEXT_TRACK = 3
GESTURE_STATE_PREVIOUS_TRACK = 4

gesture_state = GESTURE_STATE_NONE
prev_gesture_state = GESTURE_STATE_NONE

volume_threshold = 100
track_threshold = 150

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

        if lmList != []:
            x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
            x2, y2 = lmList[8][1], lmList[8][2]  # Index finger
            x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger
            x4, y4 = lmList[20][1], lmList[20][2]  # Pinky finger


            dist_thumb_index = hypot(x2 - x1, y2 - y1)
            dist_thumb_middle = hypot(x3 - x1, y3 - y1)
            dist_thumb_pinky = hypot(x4 - x1, y4 - y1)

            cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 13, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x4, y4), 13, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(img, (x1, y1), (x3, y3), (255, 0, 0), 3)
            cv2.line(img, (x1, y1), (x4, y4), (255, 0, 0), 3)


            prev_gesture_state = gesture_state


            if dist_thumb_middle < dist_thumb_index:
                if dist_thumb_middle <= track_threshold:
                    gesture_state = GESTURE_STATE_NEXT_TRACK
                else:
                    gesture_state = GESTURE_STATE_NONE
            else:
                if dist_thumb_index > volume_threshold:
                    gesture_state = GESTURE_STATE_VOLUME
                else:
                    gesture_state = GESTURE_STATE_NONE


            if gesture_state == GESTURE_STATE_VOLUME:
                vol = np.interp(dist_thumb_index, [volume_threshold, 350], [volMin, volMax])
                volbar = np.interp(dist_thumb_index, [volume_threshold, 350], [400, 150])
                volper = np.interp(dist_thumb_index, [volume_threshold, 350], [0, 100])
                volume.SetMasterVolumeLevel(vol, None)
                if prev_gesture_state == GESTURE_STATE_MUTE:
                    volume.SetMute(False, None)


            if gesture_state == GESTURE_STATE_MUTE:
                volume.SetMute(True, None)


            if gesture_state == GESTURE_STATE_NEXT_TRACK:
                cv2.putText(img, "NEXT track", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


            if dist_thumb_pinky < dist_thumb_index:
                if dist_thumb_pinky <= track_threshold:
                    gesture_state = GESTURE_STATE_PREVIOUS_TRACK


            if gesture_state == GESTURE_STATE_PREVIOUS_TRACK:
                cv2.putText(img, "PREV track", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
            cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
