# Assignment for Week 4
## np.mean(data, axis=0)用法
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
