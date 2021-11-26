# 21.11.26 validation4. Validation3에서 했던 range를 train_test_split으로 train/test/val로 각각 10개 3개 3개로 슬라이싱 하여 진행해 본다.
# Validation3.py에서 했던 방법은 test_split을 두 번을 쓰는 건데 이러면 불편할 수 있으니까 하나를 뺀다

# 대신 #3.fit에서, validation data부분을 잘라 줄 수 있다.validation_split=0.3
from sklearn.utils import validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터
# Train_set
x = np.array(range(1,17))
y = np.array(range(1,17))

# Train&test&val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8125, shuffle=True, random_state=66)
# 13개, 3개로 나누었다.                                     #비율 나누는법 =?? 계산기??


# 확인용
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# print(x_val.shape)
# print(y_val.shape)

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
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.3)
# Train_data 의 30%를 Validation으로 사용하겠다.

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( [17] )
print("17의 예측값 : ", y_predict)

# < Tuning Result >
# model.add(Dense(4, input_dim=1))
# model.add(Dense(7))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))
# epochs=100, batch_size=1
# loss :  6.555941206576321e-11
# 17의 예측값 :  [[16.999994]]
# 16.999994로 유의미한 값을 가진다.

