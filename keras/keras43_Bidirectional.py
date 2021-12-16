# - Bidirectional -
# LSTM의 단방향적 문제를 시간지남에 따라 데이터가 소실되는 부분이 생기지만
# 반대편으로 양방향으로 진행시 가중치가 더해지기에 성능이 좋을 것이다.란 이론
# 어떤 RNN을 사용할지 명시해줘야 함. => wrapping
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional

# 1. 데이터
x = np.array(
            [[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5],
             [4, 5, 6]]
            )
y= np.array( [4, 5, 6, 7] )

print(x.shape, y.shape) # (4, 3) (4, )

x = x.reshape(4, 3, 1)

# 2. 모델 구성

model = Sequential()
# model.add( SimpleRNN(32,input_shape=(3, 1), return_sequences=True) ) 
# model.add(Bidirectional(SimpleRNN(10))) #input값 까지 왔다갔다 하면안되니까(정해진거니까 반대편에서도 같은 인풋이 들가면 안되니까) 래핑하는 Bidirectional은 SimpleRNN을 따로 선언 하고 
#                       위의 두줄 = 혹은 아래 한 줄
model.add(Bidirectional(SimpleRNN(10),input_shape=(3, 1)))

model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
model.summary()
'''
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

# 4. 평가, 예측
model.evaluate(x, y)
result = model.predict( [[[5], [6], [7]]] )
print(result)

'''