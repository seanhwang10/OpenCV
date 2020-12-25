# OpenCV commands 

OpenCV 커맨드 정리 

md editor/viewer: [Typora](https://typora.io/)  

# 1. Basics

- [x, y] = width, height 

  - 우로 갈수록 x increase 

  - #### 하로 갈수록 y increase  

## 이미지 import 

```
img = cv2.imread("path/image.png") 
```
## 이미지 display 
```python
cv2.imshow("name of window", img)
cv2.waitKey(딜레이) in ms 
```
* 딜레이 0 = 무한딜레이, 이미지 항상 보임  

## 비디오 import 
* 비디오 캡쳐를 만들어야함 
```cap = cv2.VideoCapture("path/video.mp4") ```

## 비디오 display 
* 비디오는 이미지의 연속이라 루프 필요 
```python
while True: 
	success, img = cap.read() 
	cv2.imshow("name of window", img)
```
* break 문 만들기 
```python
if cv2.waitKey(1) & 0xFF == ord('q'): 
	break 
```
* 키보드 Q 눌렀을때 break 

## 웹캠 import 하기 
* 비디오랑 비슷함 
```python
cap = cv2.VideoCapture(0) 
```
* 웹캠 하나면 그냥 0 하면 됨 
```python
cap.set(3,640) //width ID:3
cap.set(4,480) //height ID:4 
cap.set(10,100) //brightness ID:10 
```
# 2. OpenCV Functions 

## Grayscale 하기 
```python
imgGray = cv2.cvtColor(img, cv2.COLOR_..) 
convention: COLOR_BGR2GRAY
```
## Blur 
```python
imgBlur = cv2.GaussianBlur(img, (3,3),0)
```
* (3,3) = kernalsize. 얼마나 blur 할건지 
* Odd number 여야함. 클수록 더 blur 

## Edge detector 
```python
imgCanny = cv2.Canny(img,200,200)
```
* Threshold 값 높일수록 더 둔감.

## Erosion, Dilation 컨셉 



# 3. Resizing 

## 사이즈 알아내기 img.shape
```python
print(img.shape)
```
- OUTPUT
``` python
(838, 682, 3)
img.shape[0] = 838 //Height 
img.shape[1] = 682 //Width 
```
- (Height, Width, Num of Channel RGB) 

## Resize 
``` python
imgResize = cv2.resize(img,(300,200))
```
- (Height, Width) of image_name 설정 

## Crop 
```python
imgCrop = img[0:200, 200:500]
```
- Array 로 생각해서 [width, height] range 설정 



# 4. Shapes and Texts 

## Coloring 

- numpy 사용해서 512 x 512 black image 만들기 

``` python
img = np.zeros((512,512)) 
```

- Color functionality 줄라면 3 채널 필요 
  - Blue, Green, Red 
  - img[:] 하면 전체 다 

```python
img = np.zeros((512,512,3), np.uint8)
img[:] = 255,0,0 //B G R	  
```

- 이미지의 일부 section 만 하려면  

``` python
img[10:50, 100:300] = 100, 255, 0 
// Height, Width Range 지정해주면 됨 
```



## Creating objects 

공동으로 cv2.OBJECT() 

parameters: (img, starting point, ending point, color, thickness) 

- img 라는 그림 위에다가 그려주는 function 

### 1. Lines 

```python
cv2.line(img,(0,0),(300,300),(0,255,0),3)
```

- Starting at point (0,0) to point (300,300). Green line, 넓이 3 

### 2. Rectangles 

```python
cv2.rectangle(img,(0,0),(img.shape[1], img.shape[0]),(0,0,255))
```

- img.shape[1] 이랑 [0] 이 이미지의 우측하단 끝부분 Width 랑 Height 니까 저렇게 하면 꽉 참 

- 맨 뒤에다가 cv2.FILLED 추가하면 rectangle 안에 color fill 

### 3. Circles 

```python
cv2.circle(img,(250,250),30,(100,255,0),3)
```

- img, center point, radius, color, thickness 

### 4. Text 

```python
cv2.putText(img, "Panther", (255,255),cv2.FONT_ITALIC, 1, (255,255,0),1)
```

- img, "TEXT", location, font, size, color, thickness 



# 5. Warp perspective 

- 삐뚤어져있는 이미지 똑바로 맞추는거. 
  - 에어팟 사진 사용 

```python
pts1 = np.float32([[],[],[],[]]) //From original pic
pts2 = np.float32([[],[],[],[]]) //Transformed to 
```

- pts1 은 원래 사진에서 4개 포인트 추출 
  - 좌상, 우상, 좌하, 우하 순 
- pts2 는 추출된 4각형을 어떤 shape 으로 만들건지 edge 설정 

```python
matrix = cv2.getPerspectiveTransform(pts1,pts2)
finalImg = cv2.warpPerspective(img, matrix, (width,height))
```

- pts1, pts2 로 perspective matrix 만들어 준 뒤 
- warpPerspective 함수 이용하고 display! 

```python
width,height = 300,300
pts1 = np.float32([[178,349],[464,229],[300,621],[579,507]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
finalImg = cv2.warpPerspective(img, matrix, (width,height))

cv2.imshow("original",img)
cv2.imshow("Airpod", finalImg)
cv2.waitKey(0)
```

![](https://github.com/seanhwang10/OpenCV/blob/main/images/warp_perspective_outcome.PNG) 

![](https://github.com/seanhwang10/OpenCV/blob/main/images/warp_perspective_outcome2.PNG)



# 6. Joining Images 

- 다수의 이미지를 하나의 이미지로 stacking 하는것

```python
imgHor = np.hstack((img,img)) //horizontal stack
imgVer = np.vstack((img,img)) //vertical stack 
```

- 두개의 이미지 dimension, channel 이 같아야됨 

웹캠 2개 합치기: 

```python
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    imgHor = np.hstack((img, img))
    cv2.imshow("webcam 2", imgHor)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```



# 7. Color Detection 

- HSV = hue, saturation, value 

## Trackbars 

- Using trackbars to find optimum HSV value for a color 

```python
#Just for making the function argument valid 
def empty(a):
    pass 

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)

cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty) 
```

```
(parameter, window, initial value, max value, func) 
```

- Getting trackbar values 

```python
h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
```

```
(Trackbar parameter, Trackbar window) 
```

## Creating a mask 

- Require lower and upper array of HSV 

```python
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
```

```python
    mask = cv2.inRange(imgHSV, lower, upper)
```

- 마스크에서 원하는 색상은 WHITE 
- 안 원하는 색상은 BLACK 으로 될때까지 Trackbar 조정 
  - 최적화된 Hue, Saturation, Value의 Min Max 값 찾기 

![](https://github.com/seanhwang10/OpenCV/blob/main/images/mask_outcome.PNG)











HSV 경우 Hue, Saturation, Value 의 Min Max 필요 = 6 TBs 

```python
cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)
```

- In OpenCV, 
  - Hue Max = 179 
  - Sat Max = 255 
  - Value Max = 255











