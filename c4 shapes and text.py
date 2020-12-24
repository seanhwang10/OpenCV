import cv2
import numpy as np

# img = np.zeros((512,512,3), np.uint8)
img = cv2.imread("images/panther.PNG")
# print(img.shape)
# img[200:350, 100:400] = 255,0,0
cv2.line(img,(0,0),(300,300),(0,255,0),3)
cv2.rectangle(img,(0,0),(img.shape[1]-10, img.shape[0]-10),(0,0,255))
cv2.circle(img,(150,250),30,(100,255,0),3)
cv2.putText(img, "Panther", (100,455),cv2.FONT_ITALIC, 2, (255,255,0),1)

cv2.imshow("image",img)
cv2.waitKey(0)