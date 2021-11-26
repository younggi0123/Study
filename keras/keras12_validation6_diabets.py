# keras11에서 했던 diabets을 validation으로 진행
# 당뇨병 데이터를 이용하여, train 0.6~0.8(권장) 기준 R2 0.62 이상을 도출
# train_loss와 validation_loss의 간격을 확인
# t_l이 0.00001인데 v_l이 100이면 과적합된 데이터라 새로운 데이터를 던져줘도 튈 것이 분명함
# ★ 이는 결국, 과적합의 지표가 된다

from sklearn.utils import validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np

# 1. 데이터
# Data load
datasets = load_diabetes()
# Train_set
x = datasets.data
y = datasets.target
#print(x.shape) #feature=13
#print(y.shape) #feature= 1

# Train&test&val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=49)
#확인용
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# 2. 모델구성
# Model Set
model = Sequential()
# Model Add
model.add(Dense(16, input_dim=10))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
# Fit
model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_split=0.3, verbose=2)
# Train_data 의 30%를 Validation으로 사용하겠다.

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict(x_test)
# print("예측값 : ", y_predict)

# R2_predict
r2 = r2_score(y_test, y_predict) #y_predict test비교
print('R2스코어 : ', r2)

# < Tuning Result >
# loss :  3100.953857421875
# R2스코어 :  0.5022842307399848

#loss :  2040.6923828125
#R2스코어 :  0.6170095971921117

# 결과값 도출
# Hyper Parameter Tuning
# model.add(Dense(16, input_dim=10))
# model.add(Dense(15))
# model.add(Dense(14))
# model.add(Dense(13))
# model.add(Dense(12))
# model.add(Dense(11))
# model.add(Dense(10))
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(6))
# model.add(Dense(5))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(2))
# model.add(Dense(1))
# epochs=1000, batch_size=30
# Epoch 1000/1000(마지막 epo), 9/9 - 0s - loss: 3092.6023 - val_loss: 3070.1965
# loss :  2020.965576171875
# R2스코어 :  0.6207118851322933
# R2가 0.6207의 유의미한 값이 도출되며, train_loss와 val_loss가 큰 차이를 보이지 않는다.