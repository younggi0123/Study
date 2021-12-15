# 【       Subject : 21'. 12. 01. keras19_1~7까지의 7종류 파일 데이터에 4개지 전처리 방법을 적용해본다.       】
# ※ 전처리 및 소스에 대한 설명은 File 1. 참고 ※


# File 3. Breast Cancer Data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
import numpy as np

# 1. 데이터
# Data load
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# print(np.min(x), np.max(x))     #  0.0 4254.0
x = x.reshape(569, 30, 1)         # 트레인 셋 나누기 전에 리쉐입

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

# print(y_test[:11])              # 값찍방법 1. 값을 찍어보니 [1 1 1 1 1 0 0 1 1 1 0]로 이진분류이다. 시그모이드 ㄱㄱ
# print(np.unique(y))             # 값찍방법 2. [0 1]
print(x.shape,y.shape)                  # (569, 30)(569, )


# 2. 모델링
# Model Set
model = Sequential()
model.add(LSTM(20, input_shape=(30, 1)))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto', verbose=1, restore_best_weights=True)    

model.fit(x_train, y_train, epochs=100, batch_size=3, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# results = model.predict(x_test[:11])
# print(y_test[:11])
# print(results)


# loss: 0.2800 - accuracy: 0.9123 