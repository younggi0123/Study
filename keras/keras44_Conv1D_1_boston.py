from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (506, 13) (506, )

x=x.reshape(506,13,1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, shuffle=True, random_state=42)

#2. 모델구성
model = Sequential()
# model.add(LSTM(50, input_length=13, input_dim=1))
model.add(Conv1D(6, 2, input_shape=(13,1)))
model.add(Dense(36))
model.add(Dense(20))
model.add(Dense(11))
model.add(Dense(6))
model.add(Flatten())
model.add(Dense(3))

model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import time
start = time.time()
model.fit(x_train,y_train, epochs=300, batch_size=32)
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# 걸린시간 :  22.257 초
# loss :  30.615388870239258
# r2스코어 :  0.5891274393145453