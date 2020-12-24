import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    imgHor = np.hstack((img, img))
    cv2.imshow('webcam 2', imgHor)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


