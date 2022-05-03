# FETCHCOVTYPE으로 해본다

# LDA - LinearDiscriminantAnalysis ( 선형판별분석 )
# LDA는 y도 명시해줘야한다 (y라벨 따라가서 )
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# 1. 데이터
datasets = fetch_covtype()
# datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)      # (581012, 54)

from sklearn.datasets import fetch_openml
housing = fetch_openml(name='house_prices', as_frame=True)

# pca = PCA(n_components=13)
lda = LinearDiscriminantAnalysis()

# n_components=29를 매개변수로 넣으면 오류가 뜨는데
# ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
# n_features( 즉, x )보단 커야하고 n_classes-1 = 즉, 1개까지는 쓸 수 있다. (즉, y값 까진 인정)


# x = pca.fit_transform(x)
x = lda.fit_transform(x, y)


# print(x)
print(x.shape)  # (581012, 6)

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



# with LDA
# (581012, 54)
# (581012, 6)
# 결과 :  0.7882498730669604