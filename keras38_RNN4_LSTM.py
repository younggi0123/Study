# 통상 RNN은 LSTM임.
# ★https://cvml.tistory.com/27★
# https://limitsinx.tistory.com/62

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

# 1. 데이터
x = np.array([[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5],
             [4, 5, 6]]
             )
y= np.array( [4, 5, 6, 7] )
print(x.shape, y.shape) # (4, 3) (4, )

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행        , 열       , 몇 개씩 자르는지!!!)
x = x.reshape(4, 3, 1)


model = Sequential()
model.add(LSTM(32, activation='tanh', input_shape=(3, 1)) )  # SimpleRNN만 LSTM으로 바꿔준다.
model.add(Dense(10, activation='linear'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=700)

# 4. 평가, 예측
model.evaluate(x, y)
result = model.predict( [[[5], [6], [7]]] )
print(result)