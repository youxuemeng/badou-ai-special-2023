import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图
img =cv2.imread('lenna.png',0)
# 获取照片高度和宽度
h,w=img.shape[:]
# 二位像素转为一维
data = img.reshape(h*w,1)
#数据转换为浮点类型
data =np.float32(data)
#停止条件
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
print(criteria)
#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS
print(flags)
#K-Means聚类
compactness,labels,centers =cv2.kmeans(data,4,None,criteria,10,flags)
#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

