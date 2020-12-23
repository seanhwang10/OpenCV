#OpenCV commands 

## 이미지 import 
```
img = cv2.imread("path/image.png") 
```
## 이미지 display 
```
cv2.imshow("name of window", img)
cv2.waitKey(딜레이) in ms 
```
* 딜레이 0 = 무한딜레이, 이미지 항상 보임  

- 비디오 import 
* 비디오 캡쳐를 만들어야함 
cap = cv2.VideoCapture("path/video.mp4") 

- 비디오 display 
*비디오는 이미지의 연속이라 루프 필요 
while True: 
	success, img = cap.read() 
	cv2.imshow("name of window", img)
*break 문 만들기 
if cv2.waitKey(1) & 0xFF == ord('q'): 
	break 
*키보드 Q 눌렀을때 break 

- 웹캠 import 하기 
*비디오랑 비슷함 

cap = cv2.VideoCapture(0) 
* 웹캠 하나면 그냥 0 하면 됨 
cap.set(3,640) //width ID:3
cap.set(4,480) //height ID:4 
cap.set(10,100) //brightness ID:10 

2. OpenCV Functions 

- Grayscale 하기 
imgGray = cv2.cvtColor(img, cv2.COLOR_..) 
convention: COLOR_BGR2GRAY

- Blur 
imgBlur = cv2.GaussianBlur(img, (3,3),0)
*(3,3) = kernalsize. 얼마나 blur 할건지 
*Odd number 여야함. 클수록 더 blur 

- Edge detector 
imgCanny = cv2.Canny(img,200,200)
*Threshold 값 높일수록 더 둔감.

- Erosion, Dilation 컨셉 

3. Resizing 

- 사이즈 알아내기 
print(img.shape)


