import numpy as np
from sklearn.svm import LinearSVC # supprot vector machine
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. 데이터
# 비트연산자 "AND" 데이터 구축
#and
#  00/
#  0/1
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 0, 0, 1]

# 2. 모델
# model = LinearSVC()
model = Perceptron()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_predict)
results = model.score(x_data, y_data)
print("model.score : ", results)

acc = accuracy_score(y_data, y_predict)
print("accuracy_score : ", acc)