# LSTM를 개량한 GRU

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

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
model.add(LSTM(10, activation='linear', input_shape=(3, 1)) )
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()\
