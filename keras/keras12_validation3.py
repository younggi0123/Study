# 21.11.26 validation3. Validation1,2에서 했던 range를 train_test_split으로 train/test/val로 각각 10개 3개 3개로 슬라이싱 하여 진행해 본다.
# train_test_split은 사용방법이 쉽고 셔플까지 적용되므로 간단하고 합리적인 함수이지만 Validation Set을 따로 만들어주지는 않는다.
# 이 때, Validation Set이 필요하다면 분할한 데이터 중 한 덩어리에 해당 함수를 한 번 더 사용하면 된다.

from sklearn.utils import validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
# Train_set
x = np.array(range(1,17))
y = np.array(range(1,17))
# train_test_split이 들어간 상태에서 3등분으로 나누시오.

# Train&test&val_set

from sklearn.model_selection import train_test_split
# x, y를 x_train,y_train & x_test, y_test로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.85, shuffle=False, random_state=66)

# x_train, y_train를 x_train,y_train & x_val, y_val로 분리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, # 들어오는 데이터 유의
        test_size=0.2, shuffle=False, random_state=66)

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
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

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

# loss :  0.00019629922462627292
# 1차) 17의 예측값 :  [[16.98483]]

# loss :  6.555941206576321e-11
# 2차) 17의 예측값 :  [[16.999994]]
# 16.999994로 유의미한 값을 가진다.

