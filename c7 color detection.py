import cv2
import numpy as np

def empty(a):
    pass

img = cv2.imread("images/velostern.PNG")
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)
cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)

cv2.imshow("original", img)
cv2.imshow("HSV", imgHSV)
cv2.waitKey(0)