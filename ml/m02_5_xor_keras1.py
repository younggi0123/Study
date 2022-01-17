# 02_4를 케라스식 코딩으로 변경해본다

import numpy as np
from sklearn.svm import LinearSVC,SVC # supprot vector machine
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XOR 비트연산자
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = LinearSVC()   # 여전히 ACC 1 안 나와
# model = Perceptron()  # 여전히 ACC 1 안 나와
# model = SVC()       # SVC로 ACC 1.0 달성
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) # 0또는 1이니까

# compile gogo
# 3. 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
y_predict = model.predict(x_data)
results = model.evaluate(x_data, y_data)

print(x_data, "의 예측결과 : ", y_predict)
print("metrics_acc : ", results[1])    # [0] loss, [1] accuracy

acc = accuracy_score(y_data, np.round(y_predict,0))
print("accuracy_score : ", acc)

# acc = accuracy_score(y_data, y_predict)
# print("accuracy_score : ", acc)