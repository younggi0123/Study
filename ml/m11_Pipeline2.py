# 'P'ipeline

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Data
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)

# 원래 방법
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델구성
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

# 파이프라인 사용 방법
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.decomposition import PCA
# 벡터화해서 컬럼feature을 축소하는것(차원축소)
# 비지도 학습? y가 없는 학습일 뿐

#2. 모델
# model = SVC()
# model = make_pipeline(MinMaxScaler(), StandardScaler(), SVC())
# model = make_pipeline(MinMaxScaler(), PCA(), SVC())

###################################################################################################
#                                           수 업 내 용                                           #
###################################################################################################



model = Pipeline(  [  ("mm", MinMaxScaler()), ("svc",SVC())] ) # Pipeline사용법 : 1. 리스트 씌우기 2.이름을 명시

# Pipeline은
# make_pipeline과 Pipeline으로 두가지로 쓸 수 있다
# (이거쓰는법이 마치 merge와 concatrate 처럼 이름과 방식만 다르고 형식은 같은거임)


###################################################################################################

# 스케일링 두번해도 상관없어. 그런다고 꼭 좋아지는건 아니야.

#3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

# 1. model = make_pipeline(MinMaxScaler(), StandardScaler(), SVC())
# print(model)
# Pipeline(steps=[('minmaxscaler', MinMaxScaler()), ('svc', SVC())])
print("result : ", result)
# result :  0.975
print("accuracy-score : ", acc)
# accuracy-score :  1.0


# 2. model = make_pipeline(MinMaxScaler(), PCA(), SVC())
# result :  0.975
# accuracy-score :  1.0