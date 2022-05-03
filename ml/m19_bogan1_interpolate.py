# 결측치 처리
# 1. 행또는 열 삭제
# 2. 임의의 값
# fillna - 0, ffill, bfill, 중위값, I평균값 ... 76767임의의 값
# 3. 보간 - interpolate
# 4. 모델링 - predict
# 5. 부스팅계열 적용 - 통상 결측치, 이상치에 대해 자유롭다. (믿거나말거나)

import pandas as pd
from datetime import datetime
import numpy as np

# 날짜에 따른 결측치 임의 생성
dates = ['1/24/2022', '1/25/2022', '1/26/2022',
         '1/27/2022', '1/28/2022', ]
dates = pd.to_datetime(dates)
print(dates)

# series:vector , dataframe:행렬
ts = pd.Series( [ 1, np.nan, np.nan, 8, 10 ], index=dates)
ts = pd.Series( [ 2, np.nan, np.nan, 8, 10 ], index=dates)
print(ts)
# 2022-01-24     1.0
# 2022-01-25     NaN
# 2022-01-26     NaN
# 2022-01-27     8.0
# 2022-01-28    10.0


# ▼▼▼▼▼▼▼▼▼위의 Nan에 대한 결측치를 처리하고자 한다.!▼▼▼▼▼▼▼▼▼▼▼▼▼
# (선형회귀 법)
ts = ts.interpolate()
print(ts)
# 2022-01-24     1.000000
# 2022-01-25     3.333333
# 2022-01-26     5.666667
# 2022-01-27     8.000000
# 2022-01-28    10.000000

# 2022-01-24     2.0
# 2022-01-25     4.0
# 2022-01-26     6.0
# 2022-01-27     8.0
# 2022-01-28    10.0