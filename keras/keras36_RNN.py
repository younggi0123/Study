# RNN

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
# 4행 3열 데이터를 1개씩 잘라 훈련시켜서 RNN모델로 만들겠다. 란 소리.
# 1개 짜른게 그 다음 연산  1개 짜른게 그 다음 연산  1개 짜른게 그 다음 연산 ~~
# 예를 들어, 행 20개라면 한개한개씩 연산하며 20개까지 닿을 것이다.
# RNN의 문제점은 뒤로갈수록 연산이 소실되기에 만회하는 애들이 생기지만서도
# 앞부분의 데이터가 완벽히 뒷부분까지 연결되지 않기에(소실하기에), 이를 반영해줘야 한다.
# 쉽게 결과 값 도출 바로전의 연산 가중치의 연산의 영향이 가장 클 것이다.

print(x.shape, y.shape) # (4, 3) (4, )

# 쉽게 설명)
# input_shape = (batch_size, timesteps, feature)
# input_shape = (행,         열,        몇 개씩 자르는지!!!)
# batch_size를 행의 개수라 하는 이유 : ?

# 2차원을 3차원으로 바꾸려면 reshape. reshape는 내용물과 순서는 바뀌면 안 된다. 차원 데이터의 개수는 그대로이다.
x = x.reshape(4, 3, 1)

# 2. 모델 구성
# model = Sequential()
# model.add( SimpleRNN(32, activation='linear', input_shape=(3, 1)) )  # inputshape에 행은 안 넣어
#                                                                      # activation은 다음레이어로 가는 값을 한정시키는 것이기에 RNN의 input레이어부터 삽입 가능함
# model.add(Dense(16))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4))
# model.add(Dense(2))
# model.add(Dense(1))

model = Sequential()
model.add( SimpleRNN(32, activation='linear', input_shape=(3, 1)) )  # inputshape에 행은 안 넣어
                                                                     # activation은 다음레이어로 가는 값을 한정시키는 것이기에 RNN의 input레이어부터 삽입 가능함
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


# Output도 똑같이 2차원을 던져준다. cnn도 조각조각 낸 것이(조각낸 이미지 to 조각낸 이미지) 4차원데이터에서 4차원 output으로 이어지지만,
# rnn은 3차원에서 2차원으로 바로 엮어져서(위에서 뽑아낸 값 : 4,5,6,7의 원래 2차원 데이터) Dense와 바로 엮을 수 있다,
#                 ▼
#                INPUT   OUTPUT      비고
# DNN            2차원   2차원        X
# RNN            3차원   2차원        X
# CNN(conv2D)    4차원   4차원       FLATTEN해서 dense에 엮어줘야.
# CNN(conv1D)    3차원   

# 예를 들어, 이미지의 4차원 데이터 받은경우, CNN 으로 하면 좋겠다?! 차원을 하나 빼서 RNN으로 돌릴 수 있을까??
# 반대로 주가데이터를 받아 반대로 CNN의 Convolution으로 할 수도 있겠지?(conv1D)


# ※ np.array 설명 =>
# Pandas에 들어가 있는 수치 데이터는 numpy이고, pd는 섞어서 사용하는 것이 가능하다.
# 인공지능에서 사용하는 모든 수치는 numpy임에 유의한다.


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 평가지표는 mse든 mae든 노상관, optimizer는 loss를 최적화 한다.
# model.compile(loss='mae', optimizer='adam') # 평가지표는 mse든 mae든 노상관, optimizer는 loss를 최적화 한다.
model.fit(x, y, epochs=1000)                 # batch_default 32 니까 1epoch에 싹 다 들어갈 것.

# 4. 평가, 예측
model.evaluate(x, y)
result = model.predict( [[[5], [6], [7]]] )         # 8이 나와야 함.
print(result)

# predict에 들어갈 것도 input과 같아야 행은 다르더라도 몇개씩 자를진 나머지는 동일해야.
# (4,3)-reshape-> (4,3,1)    RNN의 INPUT_SHAPE는 3,1이다. 데이터 개수 무시하므로 None,3,1
# 5 6 7 을 주고 8을 구하라니까 [5,6,7]인 3,에서 스칼라3 벡터1개짜리를 3,1인 [[5],[6],[7]]
# 에서 다시 [[[5],[6],[7]]]인 1,3,1로..PREDICT에 1,3,1이 들어간다.PREDICT의 SHAPE도 input하는 값의 shape과 같아야 돌아가지.
# 데이터 개수는 상관없지만 같은 shape을 맞추어라. ! so, predict( [[[5],[6],[7]]])이다.

# 8 예측하기
# 결과 loss: 0.0459,        predict값 :[[8.00241]]
# 결과 loss: 6.3949e-13,    predict값[[8.]]