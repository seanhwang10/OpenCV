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
- In OpenCV, 
  - Hue Max = 179 
  - Sat Max = 255 
  - Value Max = 255

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

## Extracting a specific color 

- img 랑 mask 랑 Merge 하기위해 bitwise and 사용 

```python
imgFinal = cv2.bitwise_and(img,img,mask=mask)
```

```
(src1, src2, mask)
```

![](https://github.com/seanhwang10/OpenCV/blob/main/images/mask_final_outcome.PNG)

- 최종 코드 

```python
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
```

# 8. Shape Detection 

- Gray -> Blur -> Canny 순으로 변환 
- Contours 찾는 function 
  - cv2.RETR_EXTERNAL = retreive external 
  - External extreme coutour 찾는 방법임. 

```python
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 1000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True) //Closed Shape 이면 TRUE
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            # print(len(approx))
            objCor = len(approx) //몇각형인지 

```

- Bounding box 

```python
x, y, w, h = cv2.boundingRect(approx)
cv2.rectangle(imgContour, (x,y),(x+w,y+h),(0,255,0),4)
```

## Shape identifying algorithm 

- objCor 사용. objCor 는 각 object의 corner 가 몇개인지 지정.
- Object type 을 찾은 뒤 Bounding box 입히고 text 출력까지 시키는게 목표. 

```python
if objCor == 3:
    objectType = "Triangle"
elif objCor == 4:
    aspRatio = w/float(h)
    if aspRatio > 0.95 and aspRatio < 1.05:
        objectType = "Square"
    else:
        objectType = "Rectangle"
elif objCor > 4 and objCor < 8:
    objectType = "N-tagon"
else:
    objectType = "Circle"
cv2.putText(imgContour, objectType, (x+w+5,y+h),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))

```

![](https://github.com/seanhwang10/OpenCV/blob/main/images/shape_output.PNG)

```python
import cv2
import numpy as np

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 1000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True) #True if you want closed shape
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            # print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x,y),(x+w,y+h),(0,255,0),4)

            if objCor == 3:
                objectType = "Triangle"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4 and objCor < 8:
                objectType = "N-tagon"
            else:
                objectType = "Circle"
            cv2.putText(imgContour, objectType, (x+w+5,y+h),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))


img = cv2.imread("images/shapes.PNG")
imgContour = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)

getContours(imgCanny)

imgStack = stackImages(0.6, ([img, imgGray], [imgCanny, imgContour]))
cv2.imshow("Shapes", imgStack)
cv2.imshow("Detect", imgContour)
cv2.waitKey(0)
```

![](https://github.com/seanhwang10/OpenCV/blob/main/images/shape_output2.PNG)

- Can improve the accuracy & details by adding more conditions in the function above. 



# 9. Facial Detection & Object Recognition 

- 찾으려는 형상 (얼굴, 물체)등을 학습시켜 cascade 만들어야함. 
  - OpenCV provided cascade 로 사람 얼굴 identify 는 가능. 
  - 누군지 identify 하려면 내가 custom cascade 만들어야함

Haarcascade 사용. 

```python
faceCascade = cv2.CascadeClassifier("CASCADE위치")
```

- Grayscale 로 바꾼 img 에서: 

```python
faces = faceCascade.detectMultiScale(imgGray, 2, 4)
```

```
(source image, sensitivity, parameter2) 
```

- 얼굴 인식 잘 될때까지 두개 파라미터 수정. 
- faceCascade.detectMultiscale 하면 x,y,w,h 나옴 
  - Bounding Box 그리는대 사용. 

```python
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
```

- 사람, 고양이 얼굴인식 (웹캠) 
  - Cascade 1: haar frontalface default 
  - Cascade 2: Frontal cat face 

```python
import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
catCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalcatface_extended.xml")

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.5, 4)
    catface = catCascade.detectMultiScale(imgGray, 1.1, 4)

    cap.set(10, 100)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,"Saram",(x+w+5,y+h),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))

    for (x,y,w,h) in catface:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img,"goyangE",(x+w+5,y+h),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))



    cv2.imshow("Faces", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

```

![](https://github.com/seanhwang10/OpenCV/blob/main/images/catrecog.gif)

- Still Image

```python
import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
catCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalcatface_extended.xml")

img = cv2.imread("images/catandhuman.PNG")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(imgGray, 1.5, 4)
catface = catCascade.detectMultiScale(imgGray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img,"Human",(x+w+5,y+h),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))

for (x,y,w,h) in catface:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.putText(img,"Cat",(x+w+5,y+h),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))

cv2.imshow("Result", img)
cv2.waitKey(0)

```

![](https://github.com/seanhwang10/OpenCV/blob/main/images/catandhuman.PNG)

![](https://github.com/seanhwang10/OpenCV/blob/main/images/face_outcome.PNG)



# 10. Facial Recognition 

- 사람 얼굴을 알아보고 학습된 cascade 통하여 이름, 성별 등 구별. 

```python

```





# 11. Training Faces via Machine Learning 

- 사진과 이름을 주고 Train. 그리고 identify 시키기. 
  - 딥러닝 활용: Tensorflow, dlib 참조
  - 여기서는 face-detection 라이브러리 사용 
- 12/27



# Resources 

## Image Stacking Function 

```python
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
```

- 호출: 

``` 
imgStack = stackImages(scale, image array)
```

```python
imgStack = stackImages(0.7,([img,imgHSV],[mask,imgFinal]))
cv2.imshow("Stacked", imgStack)
```



## Shape identification 

```python
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 1000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True) #True if you want closed shape
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            # print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x,y),(x+w,y+h),(0,255,0),4)

            if objCor == 3:
                objectType = "Triangle"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4 and objCor < 8:
                objectType = "N-tagon"
            else:
                objectType = "Circle"
            cv2.putText(imgContour, objectType, (x+w+5,y+h),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
```



