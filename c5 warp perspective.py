import cv2
import numpy as np

img = cv2.imread("images/bottle.PNG")

# 554,308
# 607,270
# 621,409
# 677,371

width,height = 200,400
pts1 = np.float32([[554,308],[607,270],[621,409],[677,371]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
finalImg = cv2.warpPerspective(img, matrix, (width,height))

cv2.imshow("original",img)
cv2.imshow("sign", finalImg)
cv2.waitKey(0)