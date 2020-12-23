import cv2
import numpy as np

img = cv2.imread("images/panther.PNG")
print(img.shape)

# imgResize = cv2.resize(img,(1000,500))
imgCrop = img[0:200, 200:500]

cv2.imshow("cat",imgCrop)
cv2.waitKey(0)