# Assignment for Week 5
## np.insert()用法
numpy.insert可以有三个参数（arr，obj，values），也可以有4个参数（arr，obj，values，axis）  
arr – 是一个数组，可以是一维的也可以是多维的，在arr的基础上插入元素  
obj – 是元素插入的位置  
values – 是需要插入的数值  
axis – 是指示在哪一个轴上对应的插入位置进行插入  
## cv2.GaussianBlur()用法
语法：GaussianBlur（src，ksize，sigmaX [，dst [，sigmaY [，borderType]]]）-> dst  
**例：**
```python
dst = cv2.GaussianBlur(src, (5, 5), 0)
```
src – 输入图像；图像可以具有任意数量的通道，这些通道可以独立处理，但深度应为CV_8U，CV_16U，CV_16S，CV_32F或CV_64F。  
dst – 输出图像的大小和类型与src相同。  
ksize – 高斯内核大小。 ksize.width和ksize.height可以不同，但​​它们都必须为正数和奇数，也可以为零，然后根据sigma计算得出。  
sigmaX – X方向上的高斯核标准偏差。  
sigmaY – Y方向上的高斯核标准差；如果sigmaY为零，则将其设置为等于sigmaX；如果两个sigmas为零，则分别从ksize.width和ksize.height计算得出；为了完全控制结果，而不管将来可能对所有这些语义进行的修改，建议指定所有ksize，sigmaX和sigmaY。  
## cv2.dilate()用法
语法：cv2.dilate(img, kernel, iteration)  
img – 目标图片  
kernel – 进行操作的内核，默认为3×3的矩阵  
iterations – 膨胀次数，默认为1  
**相应代码**
```python
import cv2
import numpy as np
img = cv2.imread('img.jpg', 0)
kernel = np.ones((4, 4), np.uint8)
img_dilate = cv2.dilate(img, kernel, iterations = 1)
```
## cv2.getStructuringElement()用法
语法：cv2.getStructuringElement(shape， ksize)  

shape - 代表形状类型  
* cv2. MORPH_RECT：矩形结构元素，所有元素值都是1
* cv2. MORPH_CROSS：十字形结构元素，对角线元素值都是1
* cv2. MORPH_ELLIPSE：椭圆形结构元素  

ksize - 代表形状元素的大小  
```python
import cv2
import numpy as np

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print(kernel1)
print('\n', kernel2)
print('\n', kernel3)

# 输出为：
# [[1 1 1 1 1]
#  [1 1 1 1 1]
#  [1 1 1 1 1]
#  [1 1 1 1 1]
#  [1 1 1 1 1]]
# 
#  [[0 0 1 0 0]
#  [0 0 1 0 0]
#  [1 1 1 1 1]
#  [0 0 1 0 0]
#  [0 0 1 0 0]]
# 
#  [[0 0 1 0 0]
#  [1 1 1 1 1]
#  [1 1 1 1 1]
#  [1 1 1 1 1]
#  [0 0 1 0 0]]
```
## cv2.Canny()用法
1. 使用高斯模糊，去除噪音点（cv2.GaussianBlur）
2. 灰度转换（cv2.cvtColor）
3. 使用sobel算子，计算出每个点的梯度大小和梯度方向
4. 使用非极大值抑制(只有最大的保留)，消除边缘检测带来的杂散效应
5. 应用双阈值，来确定真实和潜在的边缘
6. 通过抑制弱边缘来完成最终的边缘检测

语法：cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]]) 

## cv2.findContours()用法
详解：<https://www.cnblogs.com/wojianxin/p/12602490.html>

contours, hierarchy = cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])  
image - 第一个参数是寻找轮廓的图像  
mode - 表示轮廓的检索模式，有四种（这里说的都是cv2接口，因为cv2接口与cv3接口不同）
* cv2.RETR_EXTERNAL：表示只检测外轮廓
* cv2.RETR_LIST：检测的轮廓不建立等级关系
* cv2.RETR_CCOMP：建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
* cv2.RETR_TREE：建立一个等级树结构的轮廓。

method - 为轮廓的近似办法
* cv2.CHAIN_APPROX_NONE：存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
* cv2.CHAIN_APPROX_SIMPLE：压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
* cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法

## 透视变换
<img width="1081" alt="image" src="https://github.com/tangjunhao518/badou-ai-special-2023/assets/93815985/21bef868-ef21-41ab-84c6-ece7bb4359a8">
