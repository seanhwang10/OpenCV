import cv2
import numpy as np

img = cv2.imread("images/panther.PNG")
print(img.shape)


cv2.imshow("cat",img)
cv2.waitKey(0)