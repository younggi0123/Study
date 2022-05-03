# 차원을 줄였을 때 성능이 향상되는 것은 아니지만 방대한 데이터를 상대로 어느정도 자원을 덜 쓸수 있다.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터
datasets= load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape)              # load_breast_cancer = (569,30)

from sklearn.datasets import fetch_openml
housing = fetch_openml(name='house_prices', as_frame=True)

pca = PCA(n_components=14)
x = pca.fit_transform(x)
print(x.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
# [9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
#  8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
#  3.44135279e-07 1.86018721e-07]
print(sum(pca_EVR))     # 0.9999998946838411 => 1

# If a numbur of n_components are 10
cumsum = np.cumsum(pca_EVR)
# print(cumsum)
# [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989]


# If a numbur of n_components are 30
# print(cumsum)   # 14 개만 압축해도 동일한 성능이 나와야 한단 소리이다.
# [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        ]
# 결과 :  0.7178014440224423

# If a numbur of n_components are 14
# print(cumsum)  
# [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999]
# 결과 :  0.7088915995419136
# 14개일때 30개일때랑 비슷하다는 것을 알 수 있다. (컬럼 축소로 낭비를 막는다)

import matplotlib.pyplot as plt
plt.plot(cumsum)
# plt.plot(pca_EVR)
plt.grid()
plt.show()

'''
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state=66, shuffle=True
)

# 2. 모델구성
from xgboost import XGBRegressor
model = XGBRegressor()

# 3. 훈련
model.fit(x_train, y_train, eval_metric='error')

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


# without PCA(n_components=8)
# (506, 13)
# 결과 :  0.9221188601856797
    
# with PCA(n_components=8)
# (506, 13)
# 결과 :  0.7856968255504542

# import sklearn as sk
# print(sk.__version__)   # 0.24.2

'''