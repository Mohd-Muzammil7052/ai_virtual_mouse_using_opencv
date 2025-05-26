import cv2
import mediapipe as mp
import numpy as np
import random
import hand_tracking_module as htm
import pyautogui
from pynput.mouse import Button, Controller

cap = cv2.VideoCapture(0)

mouse = Controller()
detector  = htm.handDetector(detconf=0.6,trackconf=0.7)
screen_w, screen_h = pyautogui.size()

def move_mouse(index_tip):
    if index_tip is not None:
        x = int(index_tip[0]*screen_w)
        y = int(index_tip[1]*screen_h)
        pyautogui.FAILSAFE = False
        pyautogui.moveTo(x,y)

def detect_gestures(frame,lmlist,angle_ind,angle_mid,dist_thumb):
    if len(lmlist) >= 21:
        index_tip_raw = [lmlist[8][1], lmlist[8][2]]  # raw pixel coordinates
        f_h, f_w = frame.shape[0], frame.shape[1]

        # Normalize to [0, 1]
        index_tip = [
            index_tip_raw[0] / f_w,
            index_tip_raw[1] / f_h
        ]

        if dist_thumb < 50 and angle_ind > 150:
            # print(dist_thumb, angle_ind)
            move_mouse(index_tip)

        # left click
        elif dist_thumb > 50 and angle_ind < 150 and angle_mid > 150:
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame,"Left Click",(50,50),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),3)

        # right click
        elif dist_thumb > 50 and angle_ind > 50 and angle_mid < 150:
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame,"Right Click",(50,50),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),3)

        # double click
        elif dist_thumb > 50 and angle_ind < 150 and angle_mid < 150:
            pyautogui.doubleClick()
            cv2.putText(frame,"Double Click",(50,50),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),3)

        # screeshot
        elif dist_thumb < 50 and angle_ind < 150 and angle_mid < 150:
            ss = pyautogui.screenshot()
            label = random.randint(1,1000)
            ss.save(f'my_screenshot_{label}.png')
            cv2.putText(frame,"Screen Shot",(50,50),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),3)

def calculate_angle(p1,p2,p3):
    radians = np.arctan2(p3[1] - p2[1],p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1],p1[0] - p2[0])
    angle = np.abs(np.degrees(radians))
    return angle

def calculate_distance(p1, p2):
    length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    max_distance = np.hypot(640, 640)  # Example screen size diagonal
    length = np.interp(length, (0, max_distance), (0, 1000))
    return length



while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(640,640))
    f_h = frame.shape[0]
    f_w = frame.shape[1]

    frame = detector.handdetect(frame)
    lmlist = detector.findpos(frame,draw=False) 

    if len(lmlist) != 0:

        # for index finger
        x1,y1 = lmlist[8][1],lmlist[8][2]
        p1 = [x1,y1]
        x2,y2 = lmlist[6][1],lmlist[6][2]
        p2 = [x2,y2]
        x3,y3 = lmlist[5][1],lmlist[5][2]
        p3 = [x3,y3]

        angle_index = calculate_angle(p1,p2,p3)
        # print(angle_index)
        
        # for middle finger
        x4,y4 = lmlist[12][1],lmlist[12][2]
        p4 = [x4,y4]
        x5,y5 = lmlist[10][1],lmlist[10][2]
        p5 = [x5,y5]
        x6,y6 = lmlist[9][1],lmlist[9][2]
        p6 = [x6,y6]

        angle_mid = calculate_angle(p4,p5,p6)

        # for thumb
        x7,y7 = lmlist[4][1],lmlist[4][2]
        p7 = [x7,y7]
        x8,y8 = lmlist[5][1],lmlist[5][2]
        p8 = [x8,y8]

        length = calculate_distance(p7,p8)

        detect_gestures(frame,lmlist,angle_index,angle_mid,length)
    
    cv2.imshow("Image",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
