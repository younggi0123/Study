
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






################# 결측치 위치 확인
# 데이터 수가 적을경우
print(data.isnull())
#        a      b      c      d
# 0  False  False   True   True
# 1   True  False  False  False
# 2   True   True   True   True
# 3  False  False  False  False
# 4  False   True  False   True

# 데이터 수가 많을 경우 -> column별
print(data.isnull().sum())
# a    2
# b    2
# c    2
# d    3

# Info  사용
print(data.info())
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   a       3 non-null      float64
#  1   b       3 non-null      float64
#  2   c       3 non-null      float64
#  3   d       2 non-null      float64






################# 결측치 삭제=> nan이 있는 애들 삭제 됨
# axis =0 으로 행을 삭제하겠다. & axis=1찍으면 empty로 뜸
# print(data.dropna())
# print(data.dropna(axis=0))
# print(data.dropna(axis=1))

#      a    b    c    d
# 3  8.0  8.0  8.0  8.0

# None
# Empty DataFrame
# Columns: []
# Index: [0, 1, 2, 3, 4]





################# 2. 특정값

# 2.1. 각 컬럼별 평균
means = data.mean()
print(means)
# a    6.666667
# b    4.666667
# c    7.333333
# d    6.000000
data1 = data.fillna(means)
print(data1)
#            a         b          c    d
# 0   2.000000  2.000000   7.333333  6.0
# 1   6.666667  4.000000   4.000000  4.0
# 2   6.666667  4.666667   7.333333  6.0
# 3   8.000000  8.000000   8.000000  8.0
# 4  10.000000  4.666667  10.000000  6.0

# 2.2. 각 컬럼별 중위값
meds = data.median()
print(meds)
# a    8.0
# b    4.0
# c    8.0
# d    6.0
data2 = data.fillna(meds)
print(data2)
#       a    b     c    d
# 0   2.0  2.0   8.0  6.0
# 1   8.0  4.0   4.0  4.0
# 2   8.0  4.0   8.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0

# 2.3. 특정값 - front fill (ffill), back fill(bfill)
# 2.3.1. ffill - 앞의 값으로 가져온다.
data2 = data.fillna(method = 'ffill')
print(data2)
#       a    b     c    d
# 0   2.0  2.0   NaN  NaN
# 1   2.0  4.0   4.0  4.0
# 2   2.0  4.0   4.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0

# 2.3.2. bfill - 뒤의 값으로 가져온다.
data2 = data.fillna(method='bfill')
print(data2)
#       a    b     c    d
# 0   2.0  2.0   4.0  4.0
# 1   8.0  4.0   4.0  4.0
# 2   8.0  8.0   8.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 그렇다면, 본데이터의 1열의 2행, 3행 같은 nan들을 같은수로 다 채워버린다면, 데이터가 많을 땐 괜찮은 걸까??
# limit 파러미터=1를 통하여 앞에서 한 줄만 카피 하는 것으로 정해주었다.
data2 = data.fillna(method='ffill', limit=1)
print(data2)
#       a    b     c    d
# 0   2.0  2.0   NaN  NaN
# 1   2.0  4.0   4.0  4.0
# 2   NaN  4.0   4.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0

# limit 파러미터=1를 통하여 뒤에서 한 줄만 카피 하는 것으로 정해주었다.
data2 = data.fillna(method='bfill', limit=1)
print(data2)
#       a    b     c    d
# 0   2.0  2.0   4.0  4.0
# 1   NaN  4.0   4.0  4.0
# 2   8.0  8.0   8.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 2.3.3. 특정값 - 채우기
data2 = data.fillna(777)
print(data2)
#        a      b      c      d
# 0    2.0    2.0  777.0  777.0
# 1  777.0    4.0    4.0    4.0
# 2  777.0  777.0  777.0  777.0
# 3    8.0    8.0    8.0    8.0
# 4   10.0  777.0   10.0  777.0

