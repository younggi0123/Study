# GRU로 구현
# 성능이 유사할 경우 - fit에 time걸어서 속도 확인
# time으로 LSTM과 시간 비교

import time                                                                                      
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Dropout, SimpleRNN, LSTM, GRU

# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12],
              [20, 30, 40], [30, 40, 50], [40, 50, 60]]
            )
y= np.array( [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70] )
print(x.shape, y.shape) # (13, 3) (13, )

x = x.reshape(13, 3, 1)

model = Sequential()
model.add(GRU(32, activation='tanh', input_shape=(3, 1)) )
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()

model.fit(x, y, epochs=1000)

end = time.time() - start

print("걸린시간 : ", round(end),"sec")


# 4. 평가, 예측
model.evaluate(x, y)
result = model.predict( [[[77], [78], [79]]] )
print(result)