import cv2
import numpy as np

img = cv2.imread("images/airpod.PNG")

# 178,349
# 464,229
# 300,621
# 579,507

width,height = 300,300
pts1 = np.float32([[178,349],[464,229],[300,621],[579,507]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
finalImg = cv2.warpPerspective(img, matrix, (width,height))

cv2.imshow("original",img)
cv2.imshow("Airpod", finalImg)
cv2.waitKey(0)