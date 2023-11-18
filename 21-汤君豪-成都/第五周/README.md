# Assignment for Week 5
## np.insert()用法
numpy.insert可以有三个参数（arr，obj，values），也可以有4个参数（arr，obj，values，axis）  
第一个参数arr是一个数组，可以是一维的也可以是多维的，在arr的基础上插入元素  
第二个参数obj是元素插入的位置  
第三个参数values是需要插入的数值  
第四个参数axis是指示在哪一个轴上对应的插入位置进行插入  
## cv2.GaussianBlur()用法
语法：GaussianBlur（src，ksize，sigmaX [，dst [，sigmaY [，borderType]]]）-> dst  
**例：**
```python
dst=cv2.GaussianBlur(src, (5, 5), 0)
```
——src输入图像；图像可以具有任意数量的通道，这些通道可以独立处理，但深度应为CV_8U，CV_16U，CV_16S，CV_32F或CV_64F。  
——dst输出图像的大小和类型与src相同。  
——ksize高斯内核大小。 ksize.width和ksize.height可以不同，但​​它们都必须为正数和奇数，也可以为零，然后根据sigma计算得出。  
——sigmaX X方向上的高斯核标准偏差。  
——sigmaY Y方向上的高斯核标准差；如果sigmaY为零，则将其设置为等于sigmaX；如果两个sigmas为零，则分别从ksize.width和ksize.height计算得出；为了完全控制结果，而不管将来可能对所有这些语义进行的修改，建议指定所有ksize，sigmaX和sigmaY。  
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
## 透视变换
<img width="1081" alt="image" src="https://github.com/tangjunhao518/badou-ai-special-2023/assets/93815985/21bef868-ef21-41ab-84c6-ece7bb4359a8">
