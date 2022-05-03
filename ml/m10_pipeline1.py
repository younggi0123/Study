# 파이프라인 (근데, 스케일링을 가미한.) # 스케일링 자꾸 빼먹는데 주의해~
# 왜 파이프라인을 할 때의 스케일링과 차이가 있을까? 는 알아서 찾아봐~

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
from sklearn.svm import SVC


# 파이프라인 사용 방법
from sklearn.pipeline import make_pipeline, Pipeline


#2. 모델
# model = SVC()
model = make_pipeline(MinMaxScaler(), SVC())
# gridSearch에 넣어서 하이퍼 파라미터 튜닝 ㄱㄱ해보기~
# 스케일링 범위폭이 make_pipeline서 MinMaxScaler로 정해지니까 test를 따로 안 해도 정해져 있어

#3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print(model)
# Pipeline(steps=[('minmaxscaler', MinMaxScaler()), ('svc', SVC())])
print("result : ", result)
# result :  0.975
print("accuracy-score : ", acc)
# accuracy-score :  1.0