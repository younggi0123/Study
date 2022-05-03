# 회귀모델을 LDA해본다
# -> 회귀 데이터를 모두 집어넣고 LDA3와 동일하게 만들기
# -> 회귀는 n_components 직접 집어 넣어주어야 한다.

# 보스톤 집값, 당뇨병, 켈리포니아 데이터

import numpy as np

from sklearn.datasets import load_boston, load_diabetes
from sklearn.datasets import fetch_california_housing

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings(action='ignore')



# 1. 데이터
# datasets = load_boston()
# datasets = load_diabetes()
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target



# boston, california 안 되는이유
# 벨류에러
# https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown
# 해결방법
# 방법 1.
# y = y.astype("int")
# 방법 2.
#혹은 y = np.round(y,0)
# 방법 3.
#혹은 y= y*10

#################################################################################
# 방법 3에서 또한,
# 하단 참고 !!!!!!!!!!# 하단 참고 !!!!!!!!!!# 하단 참고 !!!!!!!!!!# 하단 참고 !!!

# 하지만 위와 같이 정수만 남기고 소수를 다 버려버린다면,
# 예를 들어 1.0020 1.102 1.203 1.010 1.233 1.302
# 이런 데이터가 있다면 정수만 남긴다면 1만 남는 엄청난 비효율 데이터가 될 것이다.
# 그렇기에 본 데이터가 소수점 몇 자리까지 대부분 포진되어 있는지를 파악하고자 한다.

# b = []
# for i in y:
#     b.append(len(str(i).split('.')[1]))
# print(np.unique(b, return_counts=True))

# 보스톤
# (array([1]), array([506], dtype=int64))

# 켈리포니아
# (array([1, 2, 3, 5]), array([  693,  1972, 17006,   969], dtype=int64))
# 소수 첫째자리 693, 둘째 1972, 셋째 17006, 넷째 969
# 하여, 소수 셋째자리까지 대부분 포진되어 있기에
# y= y*1000 까지 살려주는 것으로 한다.

y = y*1000
#################################################################################




print("회귀 모델")
print("lda 전 : ", x.shape)  # lda 전 :  (20640, 8)


# 1.1. train_test 분리
# , stratify=y   #stratify 제거
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state=66, shuffle=True
)

# 1.2. 스케일링
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)# 윗 문장이랑 합치면 fit_transform이겠죠?
x_test = scaler.transform(x_test)#test는 fit_transform하면안된다

# california_housing_data 때문에 썼음
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
model = XGBRegressor()
# model = XGBClassifier()

# 3. 훈련
#pass# model.fit(x_train, y_train, eval_metric='error')
#pass# model.fit(x_train, y_train, eval_metric='merror')
#회귀# default
model.fit(x_train, y_train)

# eval_metric은 tensorflow의 loss와 같다 # eval_metric은 tensorflow의 loss와 같다 #

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


# with LDA

# boston
# lda 전 :  (506, 13)
# lda 후 :  (404, 13)
# 결과 :  0.8606780455021774

# diabetes
# lda 전 :  (442, 10)
# lda 후 :  (353, 10)
# 결과 :  0.313354229055848

# california
# lda 전 :  (20640, 8)
# lda 후 :  (16512, 5)
# 결과 :  0.6624278208677861
