# OpenCV commands 

OpenCV 커맨드 정리 

md editor/viewer: [Typora](https://typora.io/)  

# 1. Basics

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

## 비디오 import 
* 비디오 캡쳐를 만들어야함 
```cap = cv2.VideoCapture("path/video.mp4") ```

## 비디오 display 
* 비디오는 이미지의 연속이라 루프 필요 
```
while True: 
	success, img = cap.read() 
	cv2.imshow("name of window", img)
```
* break 문 만들기 
```
if cv2.waitKey(1) & 0xFF == ord('q'): 
	break 
```
* 키보드 Q 눌렀을때 break 

## 웹캠 import 하기 
* 비디오랑 비슷함 
```
cap = cv2.VideoCapture(0) 
```
* 웹캠 하나면 그냥 0 하면 됨 
```
cap.set(3,640) //width ID:3
cap.set(4,480) //height ID:4 
cap.set(10,100) //brightness ID:10 
```
# 2. OpenCV Functions 

## Grayscale 하기 
```
imgGray = cv2.cvtColor(img, cv2.COLOR_..) 
convention: COLOR_BGR2GRAY
```
## Blur 
```
imgBlur = cv2.GaussianBlur(img, (3,3),0)
```
* (3,3) = kernalsize. 얼마나 blur 할건지 
* Odd number 여야함. 클수록 더 blur 

## Edge detector 
```
imgCanny = cv2.Canny(img,200,200)
```
* Threshold 값 높일수록 더 둔감.

## Erosion, Dilation 컨셉 



# 3. Resizing 

## 사이즈 알아내기 
```
print(img.shape)
```
- OUTPUT
``` 
(838, 682, 3)
```
- (Height, Width, Num of Channel RGB) 

## Resize 
``` 
imgResize = cv2.resize(img,(300,200))
```
- (Height, Width) of image_name 설정 

## Crop 
```
imgCrop = img[0:200, 200:500]
```
- Array 로 생각해서 [Height, Width] range 설정 



# 4. Shapes and Texts 

## Coloring 

- numpy 사용해서 512 x 512 black image 만들기 

``` python
img = np.zeros((512,512)) 
```

- Color functionality 줄라면 3 채널 필요 
  - Blue, Green, Red 
  - img[:] 하면 전체 다 

```
img = np.zeros((512,512,3), np.uint8)
img[:] = 255,0,0 //B G R	 
```

![image-20201223180514895](C:\Users\Alfonso\AppData\Roaming\Typora\typora-user-images\image-20201223180514895.png) 

- 이미지의 일부 section 만 하려면  

``` 
img[35:50, 
```





