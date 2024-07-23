import cv2
import pyautogui as pg
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# CONSTANTS
widthCam = 640
heightCam = 480
widthScreen, heightScreen = pg.size()
frameReduction = 150
smoothValue = 4
prevLocationX = 0
prevLocationY = 0
currLocationX = 0
currLocationY = 0

# WEBCAM INITIALIZATION
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, widthCam)  # -- WIDTH
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, heightCam)  # -- HEIGHT

detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]  # -- GET THE HAND IF IT IS PRESENT IN THE SCREEN
        x, y, w, h = hand['bbox']  # -- GET BOUNDING BOX
        lmList = hand["lmList"]  # -- GET THE LANDMARKS

        xIndex, yIndex, = lmList[8][:2]  # -- GET THE TIP OF INDEX FINGER
        xMiddle, yMiddle, = lmList[12][:2]  # -- GET THE TIP OF MIDDLE FINGER
        xPinky, yPinky = lmList[20][:2]  # -- GET THE TIP OF PINKY FINGER

        fingers = detector.fingersUp(hand)  # -- CHECK WHICH FINGERS ARE UP

        # MOVING MODE BOUNDING BOX
        cv2.rectangle(img, (frameReduction, frameReduction),
                      (widthCam - frameReduction, heightCam - frameReduction), (255, 0, 255), 2)

        # MOVING MODE: INDEX FINGER IS UP
        if (fingers[1] == 1) and (fingers[4] == 0):

            # CONVERT COORDINATES THROUGH INTERPOLATION
            xConverted = np.interp(xIndex, (frameReduction, widthCam - frameReduction),
                                   (0, widthScreen))
            yConverted = np.interp(yIndex, (frameReduction, heightCam - frameReduction),
                                   (0, heightScreen))

            # SMOOTHEN COORDINATE VALUES
            currLocationX = prevLocationX + (xConverted - prevLocationX) / smoothValue
            currLocationY = prevLocationY + (yConverted - prevLocationY) / smoothValue

            # MOVE MOUSE TO X AND Y COORDINATES
            pg.moveTo(widthScreen - currLocationX, currLocationY)
            cv2.circle(img, (xIndex, yIndex), 5, (255, 0, 255), cv2.FILLED)
            prevLocationX, prevLocationY = currLocationX, currLocationY  # -- UPDATE VALUES

        # RIGHT CLICK MODE: PINKY FINGER IS UP
        if (fingers[1] == 0) and (fingers[4] == 1):
            cv2.circle(img, (xPinky, yPinky), 5, (0, 255, 0), cv2.FILLED)
            pg.rightClick()

        # LEFT CLICK MODE: INDEX AND MIDDLE FINGER ARE UP
        if (fingers[1] == 1) and (fingers[2] == 1):

            # FIND THE DISTANCE BETWEEN THE INDEX AND MIDDLE FINGER
            length, info, img = detector.findDistance((xIndex, yIndex), (xMiddle, yMiddle),
                                                      img, (255, 0, 255), 5)

            # CLICK IF DISTANCE IS LESSER THAN 30
            if length < 30:
                cv2.circle(img, (info[4], info[5]), 5, (0, 255, 0), cv2.FILLED)
                pg.click()

    cv2.imshow("Test", img)

    # KEYBIND
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # -- 'ESC' KEY
        break

cv2.destroyAllWindows()
