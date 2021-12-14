# RNN - keras 공홈을 자주 봅시다. !
# https://keras.io/api/layers/recurrent_layers/simple_rnn/

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 1. 데이터
x = np.array([[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5],
             [4, 5, 6]]
             )
y= np.array( [4, 5, 6, 7] )
print(x.shape, y.shape)

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행,         열,        몇 개씩 자르는지!!!)
x = x.reshape(4, 3, 1)


model = Sequential()


# ★ layer의 사용방법은 다양하다 !★ layer의 사용방법은 다양하다 !★ layer의 사용방법은 다양하다 !
# model.add( SimpleRNN(10, activation='linear', input_shape=(3, 1)) )
#                               (=)
# model.add(units=10, input_shape=(3, 1))  
#                               (=)
# model.add(SimpleRNN(10, input_length=3, input_dim=1) )#     ◀───── RNN에서는 많이 쓰지만 CNN에서는 잘 쓰지 않는다.


model.add(SimpleRNN(16, input_length=3, input_dim=1) )#     ◀───── RNN에서는 많이 쓰지만 CNN에서는 잘 쓰지 않는다.
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

# 상단 케라스 공홈가면 arguments 녀석은 dense대신 'units= '로 써도 된다 적혀있엉
# https://keras.io/api/layers/recurrent_layers/simple_rnn/
# activation이 tahn(탄젠트)인 건 왜?
# input 은 [batch, timesteps, feature].

# https://keras.io/api/layers/convolution_layers/convolution2d/
# conv2D layer에서 units는 filter에 해당한다.

# https://keras.io/api/layers/core_layers/dense/
# Dense layer에서 input shape는 (batch_size, ..., input_dim) 에 해당한다
# Dense_units: Positive integer, dimensionality of the output space.


#                                           ※   DNN / RNN / CNN 정리   ※
#
#                      【◀────────────────────────────input───────────────────────────────────▶】
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────│───────────────│
#           │   Output  │       4       │       3       │       2       │           1            │   Output Shape   │       비고    │
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────│───────────────│
# Dense     │   units   │               │               │   batch(행)   │          input_dim(열) │      2           │               │
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────│───────────────│
# simpleRNN │   units   │               │   batch(행)   │  timeSteps(열)│feature(몇개씩 자르는가)│      2           │                │
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────│───────────────│
# Conv2D    │   filter  │   batch(행)   │   row(가로)   │   col(세로)   │       chanel(칼라)     │      4           │모양구성flatten │
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│
# ※ 이 때, "행 = 데이터의 개수" 이다.

# model.summary()



# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=42)

# 4. 평가, 예측
model.evaluate(x, y)
result = model.predict( [[[5], [6], [7]]] )
print(result)

# 8 예측하기
# result : 7.51