import cv2 
import numpy as np
import mediapipe as mp 
import time 

class handDetector():
    def __init__(self,mode = False, maxhands = 2, model_comp = 1, detconf = 0.5, trackconf = 0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.model_comp = model_comp
        self.detconf = detconf
        self.trackconf = trackconf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxhands,self.model_comp,self.detconf,self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils

    def handdetect(self, frame, draw = True):
        imgRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw: 
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findpos(self, frame, handNo = 0, draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myhand.landmark):
                # print(id,lm)
                h , w , c = frame.shape
                cx , cy = int(w*lm.x) , int(h*lm.y)
                lmlist.append([id,cx,cy])
                # if id == 0:
                if draw:
                    cv2.circle(frame,(cx,cy),8,(0,0,255),cv2.FILLED)
        return lmlist

def main():
    cap = cv2.VideoCapture(0)
    prevt = 0
    detector = handDetector()
    while True:
        ret, frame = cap.read()
        frame = detector.handdetect(frame)
        lmlist = detector.findpos(frame)
        
        currt = time.time()
        fps = 1/(currt - prevt)
        prevt = currt
        
        if len(lmlist) != 0 :
            print(lmlist[4])

        cv2.putText(frame,str(int(fps)) + ' FPS',(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        cv2.imshow("Image",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()