import cv2
import numpy as np

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 1000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True) #True if you want closed shape
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            # print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x,y),(x+w,y+h),(0,255,0),4)

            if objCor == 3:
                objectType = "Triangle"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4 and objCor < 8:
                objectType = "N-tagon"
            else:
                objectType = "Circle"
            cv2.putText(imgContour, objectType, (x+w+5,y+h),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgContour = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)

    getContours(imgCanny)
    cv2.imshow("Video", imgContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
