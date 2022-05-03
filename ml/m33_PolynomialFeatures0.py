import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)

# x = np.arange(10).reshape(5,2) #x1,x2 였다면? 트랜스폼하면 (5,6)


print(x)    
        # [[0 1]
        #  [2 3]
        #  [4 5]
        #  [6 7]]
print(x.shape)  # (4.2) 4행 2열
# x1 + x2상태


pf = PolynomialFeatures(degree=2)

xp = pf.fit_transform(x)
print(xp)# 변환된 데이터
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
print(xp.shape) # (4, 6)
# 한개의 컬럼이 전부 1인 x0,  x0 => 1
# x0 + x1 + x1^2 + x1*x2 + x2^2 + x2        총 6개 컬럼
# 1+ x1 + x2 + x1^2 + x1x2 + x2^2  의 6개 컬럼

# select from model을 통해 차트를 짤라낼 수 있어.
# 증폭시키고 컬럼을 짤라내고. feature importance에 따라 .
# 그때도 과적합은 주의해야.


##################################################################
x = np.arange(12).reshape(4,3)
print(x)
print(x.shape)

pf = PolynomialFeatures(degree=2)

xp = pf.fit_transform(x)
print(xp)
# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]
print(xp.shape)                     # (4, 10)
# 1 + x1 + x2 + x3
#     x1^2 + x1x2 + x1x3 +
#     x2^2 + x2x3 + x3^2

##################################################################
x = np.arange(8).reshape(4,2)
print(x)
print(x.shape)      # (4, 2)

pf = PolynomialFeatures(degree=3)
# degree하나만 늘려도 확늘어남
xp = pf.fit_transform(x)
print(xp)
# [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  1.   6.   7.  36.  42.  49. 216. 252. 294. 343.]]
print(xp.shape)                     # (4, 10)
# 1 + x1 + x2 + x1^2 + x1x2 + x2^2 + x1^3 + x1^2x2 + x1x2^2 + x2^3



# 실질적으로 degree 3까지 쓸 일은 거의 없음