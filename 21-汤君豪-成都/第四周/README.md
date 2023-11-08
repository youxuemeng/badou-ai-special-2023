# Assignment for Week 4
## np.mean(data, axis=0)用法
**代码**
```python
import numpy as np
X = np.array([[1, 2], [4, 5], [7, 8]])
print np.mean(X, axis=0, keepdims=True)
print np.mean(X, axis=1, keepdims=True)
```
**输出结果**
```python
                 [[ 1.5]
 [[ 4.  5.]]      [ 4.5]    
                  [ 7.5]]
```
**axis=0，那么输出矩阵是1行，求每一列的平均（按照每一行去求平均）；axis=1，输出矩阵是1列，求每一行的平均（按照每一列去求平均）**
**还可以这么理解，axis是几，那就表明哪一维度被压缩成1。**
