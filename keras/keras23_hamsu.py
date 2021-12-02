# 함수형 모델 # not Sequential, but Model -!

import numpy as np

# 1. 데이터
x = np.array( [range(100), range(301,401), range(1, 101)] )
y = np.array( [range(701, 801)] )
print(x.shape, y.shape)     #(3, 100) (1, 100)

x = np.transpose(x)
y = np.transpose(y)
print(x.shape, y.shape)     #(1, 10, 10, 3) (100, 2)
# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 'None by 3'을 받을거란 쉐입만 선언하고 히든레이어 쌓는다.
# 시퀀셜엔 인풋레이어 명시가 안 되어 있다./ 함수형 모델은 처음에 인풋레이어부터 명시가 된다.
# 시퀀셜은 정의, add로 레이어층을 쌓는데, 모델은 레이어층 구성 후 모델에 대한 정의를 끝에서 해준다.
# 함수형이나 시퀀셜이나 연산의 개수는 같다. 표현방식의 차이일 뿐이다.
input1 = Input( shape=(3,) )                        # Sequential과 동일한 구조로 만들겠다
dense1 = Dense(10)(input1)                          # input1 레이어로부터 받아들였다
dense2 = Dense(9, activation='relu')(dense1)
dense3 = Dense(8)(dense2)
# dense4 = Dense(8)(dense2)                           # 중간에 이런식으로 이상하게 껴넣어도 되긴함
output1 = Dense(1)(dense3)
model = Model( inputs=input1, outputs=output1 )     # 가독성 때문에 변수 네이밍을 model이라 쓴건데 원래 네이밍은 자유이다

# 함수형은 레이어까지 쌓고 나서 하는 것
# model = Sequential()
# model.add(Dense(10, input_shape=(3,)))
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(1))
# model.summary()
