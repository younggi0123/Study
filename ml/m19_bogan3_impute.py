# Impute - 대치법


import numpy as np
import pandas as pd



################# 1. 데이터 생성(결측치 포함)
data = pd.DataFrame([ [2, np.nan, np.nan, 8, 10 ],
                      [2, 4, np.nan, 8, np.nan],
                      [np.nan, 4, np.nan, 8, 10],
                      [np.nan, 4, np.nan, 8, np.nan]
                    ])
print(data.shape)       # (4,5)
data = data.transpose()
data.columns = ['a', 'b', 'c', 'd']
print(data)
#       a    b     c    d
# 0   2.0  2.0   NaN  NaN
# 1   NaN  4.0   4.0  4.0
# 2   NaN  NaN   NaN  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 실험적인 import로 iterative imputer를 쓸 수 있게 해준다.(필수 임포트)
from sklearn.experimental import enable_iterative_imputer
# import 사이킥런 - 심플인퓨터
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer, IterativeImputer

# 평균
imputer = SimpleImputer(strategy='mean')
imputer.fit(data)
data2 = imputer.fit_transform(data)
print(data2)
# 평균값으로 채워줬다.
# [[ 2.          2.          7.33333333  6.        ]
#  [ 6.66666667  4.          4.          4.        ]
#  [ 6.66666667  4.66666667  7.33333333  6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]

# 중위수
imputer = SimpleImputer(strategy='median')
imputer.fit(data)
data2 = imputer.fit_transform(data)
print(data2)
# [[ 2.  2.  8.  6.]
#  [ 8.  4.  4.  4.]
#  [ 8.  4.  8.  6.]
#  [ 8.  8.  8.  8.]
#  [10.  4. 10.  6.]]

# 최빈값
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(data)
data2 = imputer.fit_transform(data)
print(data2)
# [[ 2.  2.  4.  4.]
#  [ 2.  4.  4.  4.]
#  [ 2.  2.  4.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  2. 10.  4.]]

# 상수 값 (constant는 기본 0으로 채우지만 fill_value를 통해 원하는 값을 넣을 수 있다. & constant없이 fillvalue만 넣어도 값이 채워진다.)
imputer = SimpleImputer(strategy='constant')
imputer.fit(data)
data2 = imputer.fit_transform(data)
print(data2)
# [[ 2.  2.  0.  0.]
#  [ 0.  4.  4.  4.]
#  [ 0.  0.  0.  0.]
#  [ 8.  8.  8.  8.]
#  [10.  0. 10.  0.]]

# imputer = SimpleImputer(strategy='constant', fill_value=777)
# [[  2.   2. 777. 777.]
#  [777.   4.   4.   4.]
#  [777. 777. 777. 777.]
#  [  8.   8.   8.   8.]
#  [ 10. 777.  10. 777.]]





# Find fit & fit_transform



#################################################################

# 3. 특정컬럼단위로 적용하는 법 # 특정컬럼단위로 적용하는 법
# => 어떤 컬럼은 중위수 어떤 컬럼은 평균이여야 할 수도 있으니까.

#################################################################
# 데이터프레임 전체에 대해 적용해버려
# 컬럼별 특성에 맞게 결측치 처리방식을 주지 않았다
# (지금까지 다 All적용하였음)
# 그렇다면 컬럼별로 적용하는 법은???

# 3.1. 평균 적용
means = data['a'].mean()
print(means)
# 6.666666666666667
data['a'] = data['a'].fillna(means)
print(data)
#            a    b     c    d
# 0   2.000000  2.0   NaN  NaN
# 1   6.666667  4.0   4.0  4.0
# 2   6.666667  NaN   NaN  NaN
# 3   8.000000  8.0   8.0  8.0
# 4  10.000000  NaN  10.0  NaN


# 3.2. 중위값 적용
meds = data['b'].mean()
print(meds)
# 6.666666666666667
data['b'] = data['b'].fillna(meds)
print(data)
#            a         b     c    d
# 0   2.000000  2.000000   NaN  NaN
# 1   6.666667  4.000000   4.0  4.0
# 2   6.666667  4.666667   NaN  NaN
# 3   8.000000  8.000000   8.0  8.0
# 4  10.000000  4.666667  10.0  NaN


# 3.3. 최빈값 적용
freq = data['c'].mode()
print(freq)
# 6.666666666666667
data['c'] = data['c'].fillna(freq)
print(freq)
# 0     4.0
# 1     8.0
# 2    10.0

# 3.3. 특정값 적용
spec = data['d'].fillna(777)
data['d'] = data['d'].fillna(spec)
# data2 = data.fillna(777)


# 각 열별 특성을 따로 적용한 새로운 데이터 프레임 생성
newdata = pd.DataFrame([data['a'],
                        data['b'],
                        data['c'],
                        data['d']
                        ])
newdata = newdata.transpose()
newdata.columns = ['a', 'b', 'c','d']
# nd = newdata.loc(means)
print(newdata)





# fit에는 dataframe이 들어가는데, 우리는 컬럼만 바꾸고 싶다.
# series를 넣으면 에러가 난다.!
# 그렇다면?? How to fix
# 1개의 컬럼을 넣고싶은데 시리즈가 데이터프레임 형태로 들아면 될까?

# 3.3. 적용 -> 데이터 일부
