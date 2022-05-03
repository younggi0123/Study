# PCA 주성분분석
# autoencoder가 나오기 전까진 PCA로 압축&원위치 등 (like 기미 주근깨 제거&복구용으로 사용했다)

# 10000 by 10000이면 낭비가 심하니까 임베딩(원핫인코딩)해서 성능을 높혀 놓았다.
# 무조건 솎아내는게 아닌 압축.

import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
# 1. 데이터
datasets = load_boston()
# datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)              # load_boston's (506, 13) -> fetch_california housing's (20640, 8)

from sklearn.datasets import fetch_openml
housing = fetch_openml(name='house_prices', as_frame=True)

# n_components에 따라서 x값이 바뀌니 columns가 변화한다(본 소스에서는 13에서 8로 축소했음)
# 원래 개수보다 더 크게 증폭은 안 된다. (ex, 14~)
pca = PCA(n_components=13)
# 차원축소는 x만건드려 y는건드리지 않아. 그렇기에 y가 없어도 상관이없어
# y가 없어도 되는게 마치 비지도 학습과 비슷
x = pca.fit_transform(x)
# print(x)
# print(x.shape)        # (506, 8)

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

import sklearn as sk
print(sk.__version__)   # 0.24.2