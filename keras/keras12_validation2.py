# 21.11.26 validation2. Validation1에서 했던 range17을 train/test/val로 슬라이싱 하여 진행해 본다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터

# Train_set
x = np.array(range(17))
y = np.array(range(17))
# Train_set
x_train = x[:10]
y_train = y[:10]
# Test_set
x_test = x[11:14]
y_test = y[11:14]
# Validaiton_set
x_val = x[15:17]
y_val = y[15:17]

# 2. 모델구성
# Model Set
model = Sequential()
# Model Add
model.add(Dense(4, input_dim=1))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
# Fit
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( [17] )
print("17의 예측값 : ", y_predict)

# Tuning
# model.add(Dense(4, input_dim=1))
# model.add(Dense(7))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))
# epochs=100, batch_size=1
# loss :  9.688695712384288e-08
# 17의 예측값 :  [[16.999447]]
# 16.9994로 유의미한 값을 가진다.