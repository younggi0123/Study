# 통상 RNN은 LSTM임.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Dropout, SimpleRNN, LSTM

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
model.add(LSTM(32, activation='tanh', input_shape=(3, 1)) ) #LSTM's DEFAULT = tanh!!!!!!!!!!!!!!!
model.add(Dense(48, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
# model.summary()

# 3. 컴파일, 훈련

# Optimizer의 종류 :
# https://onevision.tistory.com/entry/Optimizer-%EC%9D%98-%EC%A2%85%EB%A5%98%EC%99%80-%ED%8A%B9%EC%84%B1-Momentum-RMSProp-Adam
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

# 4. 평가, 예측
model.evaluate(x, y)
result = model.predict( [[[77], [78], [79]]] )
print(result)
# 결과값 80을 도출하시오.
# 81.1

# 간단히 말하자면, 멀어질수록 잊혀지는 vanillaRNN의 단점을 LSTM을 사용, 망각게이트로 연관시키는 부분이다.
# (게이트 세개 state 하나 = 4개)


# CNN은 4dim to 4dim니까 레이어에 계속 붙여도 괜찮은데 RNN은 3dim to 2dim니까 동일한 레이어 두개씩 못 쓴다(내일 수업 주제)