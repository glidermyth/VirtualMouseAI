import cv2
import numpy as np
import time
import HandTrackingModule as htm
import autopy


cap = cv2.VideoCapture(0)
####################
wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
# print(wScr,hScr)
frameR = 100
smooth = 5
####################
pTime = 0
plocX, plocY = 0,0
clocX, clocY = 0,0

cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList)!= 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        # print(x1,y1,x2,y2)

        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        #only index finger : Moving mode
        if fingers[1]==1 and fingers[2]==0:
            #convert coordinates
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            # Smoothen the values
            clocX = plocX+(x3-plocX) / smooth
            clocY = plocY + (y3 - plocY) / smooth
            
            #move mouse
            autopy.mouse.move(clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX, plocY = clocX, clocY
            
        # Both index and middle fingers are up : clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between index and middle finger
            length,img, lineInfo = detector.findDistance(8,12,img)
            # print(length)
            # Clicking mouse if distance short
            if length<40:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(255,0,255),cv2.FILLED)
                autopy.mouse.click()
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

    cv2.imshow("image", img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

