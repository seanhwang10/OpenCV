import cv2

img = cv2.imread("images/panther.PNG")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img, (7,7),0)
imgCanny = cv2.Canny(img,200,200)



cv2.imshow("Original Image", img)
cv2.imshow("Gray Image", imgGray)
cv2.imshow("Edge Image", imgCanny)
cv2.waitKey(0)