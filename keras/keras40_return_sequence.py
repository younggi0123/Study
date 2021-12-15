# GRU로 구현
# return_sequences: 불리언. 아웃풋 시퀀스의 마지막 아웃풋을 반환할지, 혹은 시퀀스 전체를 반환할지 여부.

import time                                                                                      
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12],
              [20, 30, 40], [30, 40, 50], [40, 50, 60]]
            )
y= np.array( [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70] )
# x_predict = np.array()
print(x.shape, y.shape) # (13, 3) (13, )

x = x.reshape(13, 3, 1)

# 2. 모델구성
# (N, 3, 1) -> (N, 10)에서
model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(3, 1)) )    # (N, 3, 1) -> (N,3, 10) 리턴시퀀스로 연결
model.add(LSTM(32, return_sequences=True))                         # 10은 이 녀석의 Input이 되었고, 3차원을 받아야해
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(8, return_sequences=True))
model.add(LSTM(4, return_sequences=True))
model.add(LSTM(2, return_sequences=False))
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