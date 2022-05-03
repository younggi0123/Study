# 분류모델을 LDA해본다


# 아이리스, 유방암, 와인데이터, 페치코브타입

import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.datasets import fetch_covtype

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings(action='ignore')



# 1. 데이터
# datasets = load_iris()
# datasets = load_breast_cancer()
# datasets = load_wine()
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print("lda 전 : ", x.shape)  # 


# 1.1. train_test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state=66, shuffle=True, stratify=y   #stratify 균등한.
)

# stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.


# 1.2. 스케일링
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)# 윗 문장이랑 합치면 fit_transform이겠죠?
x_test = scaler.transform(x_test)#test는 fit_transform하면안된다



from sklearn.datasets import fetch_openml
housing = fetch_openml(name='house_prices', as_frame=True)

# pca = PCA(n_components=13)
lda = LinearDiscriminantAnalysis()

# n_components=29를 매개변수로 넣으면 오류가 뜨는데
# ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
# n_features( 즉, x )보단 커야하고 n_classes-1 = 즉, 1개까지는 쓸 수 있다. (즉, y값 까진 인정)


lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
# x_train = lda.fit_transform(x_train,y_train) # 이 한줄이 위 두줄과 같다.

x_test = lda.transform(x_test)


# print(x)
print("lda 후 : ", x_train.shape)  # 

# 2. 모델구성
from xgboost import XGBRegressor, XGBClassifier
# model = XGBRegressor()
model = XGBClassifier()

# 3. 훈련
# error 이진분류
# model.fit(x_train, y_train, eval_metric='error')
# merror 다중분류
model.fit(x_train, y_train, eval_metric='merror')
# default
# model.fit(x_train, y_train)

# eval_metric은 tensorflow의 loss와 같다 # eval_metric은 tensorflow의 loss와 같다 #

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


# with LDA

# iris
# lda 전 :  (150, 4)
# lda 후 :  (120, 2)
# 결과 :  1.0

# breast cancer
# lda 전 :  (569, 30)
# lda 후 :  (455, 1)
# 결과 :  0.9473684210526315

# wine
# lda 전 :  (178, 13)
# lda 후 :  (142, 2)
# 결과 :  1.0

# fetch_covtype # 얘는 n_components=5여도 가능한 부분
# lda 전 :  (581012, 54)
# lda 후 :  (464809, 6)
# 결과 :  0.7878109859470065

