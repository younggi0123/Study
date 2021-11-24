# 이후론 훈련과 테스트를 반드시 나눠 사용하게 될 것이다 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. Data
#x를 훈련용train데이터7:3테스트용test데이터  으로 갈라준다.
x_train = np.array( [1,2,3,4,5,6,7])
x_test = np.array( [8,9,10] )
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) #train data가 7개이므로 1epo당 7번씩 돌고, x와 y를 비교하므로 전체 훈련은 700번이 될 것
# batch가 2면 일괄작업 사이즈가 2이므로 x&y비교 1&1 1번, 2&2 2번 ...

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)

#1차
#loss :  3.055086851119995
#11의 예측값 :  [[8.518083]]
#2차
#loss :  0.8454137444496155
#11의 예측값 :  [[9.670679]]
#3차
#loss :  2.482920535840094e-10
#11의 예측값 :  [[11.000022]]