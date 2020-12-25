import cv2
import numpy as np

def empty(a):
    pass

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)
cv2.createTrackbar("Hue Min", "Trackbars", 105, 179, empty)
cv2.createTrackbar("Hue Max", "Trackbars", 109, 179, empty)
cv2.createTrackbar("Sat Min", "Trackbars", 54, 255, empty)
cv2.createTrackbar("Sat Max", "Trackbars", 149, 255, empty)
cv2.createTrackbar("Val Min", "Trackbars", 209, 255, empty)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

while True:
    img = cv2.imread("images/velostern.PNG")
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "Trackbars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgFinal = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow("original", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Final Image", imgFinal)
    cv2.waitKey(1)