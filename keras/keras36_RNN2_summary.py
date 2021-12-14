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
print(x.shape, y.shape)
x = x.reshape(4, 3, 1)

model = Sequential()
# model.add( SimpleRNN(10, activation='linear', input_shape=(3, 1)) )
model.add( SimpleRNN(10, activation='linear', input_shape=(3, 3)) ) #  ←
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()
# SimpleRNN   ← 부분의 결과치 parameter가 140인 이유를 찾으시오.
# 단순 계산법 아닌 왜 그런지 이해
# https://velog.io/@cateto/Deep-Learning-vanila-RNN%EC%97%90%EC%84%9C-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EC%9D%98-%EA%B0%9C%EC%88%98-%EA%B5%AC%ED%95%98%EA%B8%B0
# https://datascientist.tistory.com/25?category=961934
# Total params = recurrent_weights + input_weights + biases
# = (num_units*num_units)+(num_features*num_units) + (1*num_units)
# = (num_features + num_units)* num_units + num_units
# 결과적으로,
# ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수)
# 를 참조하여 위의 total params 를 구하면 ( ※ feature이 ( ,3)인 3인거 알지? )
# (10 * 10) + (3 * 10) + (1*10) = 140
# (노드*노드) + (노드*특성) + bias


# 설명
# 노드5, input_shape=(3, 1) 인경우.
# 3은 timestep이며, 1은 input(feature)이다.
