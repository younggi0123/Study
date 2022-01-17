# 인공지능의 겨울(암흑기) 문제 코드로 보기

import numpy as np
from sklearn.svm import LinearSVC,SVC # supprot vector machine
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. 데이터
# 비트연산자 "XOR" 데이터 구축
# => accuracy 는 죽어도 0.25~0.75 사이만 나온다..
#xor ...선을 그을 수가 없다...
#  01
#  10
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = LinearSVC()   # 여전히 ACC 1 안 나와
# model = Perceptron()  # 여전히 ACC 1 안 나와
model = SVC()       # SVC로 ACC 1.0 달성
# So, SVC는 다항식 (Polynomial) 으로 구성하였다.
# Support 머신까지는 다항식 개념이 안 들어간 선이였는데 다항식 적용으로 해결
# 이후에 이것을 신경망(다층퍼셉트론 MLP from 텐서플로우-KERAS)으로 해결

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_predict)
results = model.score(x_data, y_data)
print("model.score : ", results)

acc = accuracy_score(y_data, y_predict)
print("accuracy_score : ", acc)