# LDA - LinearDiscriminantAnalysis ( 선형판별분석 )
# LDA는 y도 명시해줘야한다 (y라벨 따라가서 )
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# 1. 데이터
datasets = load_breast_cancer()
# datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)      # 569, 30

from sklearn.datasets import fetch_openml
housing = fetch_openml(name='house_prices', as_frame=True)

# pca = PCA(n_components=13)
lda = LinearDiscriminantAnalysis()

# n_components=29를 매개변수로 넣으면 오류가 뜨는데
# ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
# n_features( 즉, x )보단 커야하고 n_classes-1 = 즉, 1개까지는 쓸 수 있다. (즉, y값 까진 인정)


# x = pca.fit_transform(x)

# LDA에서는 어차피 y를 넣어줘야함!!!!!!
# y 축소 & 
x = lda.fit_transform(x, y)


# print(x)
print(x.shape)  #569,1  shape이 완벽하게 줄어버렸다 !!!!

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state=66, shuffle=True
)

# 2. 모델구성
from xgboost import XGBRegressor, XGBClassifier
# model = XGBRegressor()
model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train, eval_metric='error')

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


# XGBoost's Default
# without PCA(n_components=8)
# (506, 13)
# 결과 :  0.9221188601856797
    
# with PCA(n_components=8)
# (506, 13)
# 결과 :  

# import sklearn as sk
# print(sk.__version__)   # 0.24.2

# with LDA
# (569, 30)
# (569, 1)
# 결과 :  0.98245614035087