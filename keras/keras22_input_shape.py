# 함수형 모델 # not Sequential, but Model -!

import numpy as np

#1. 데이터
# data 2개 이상 list
 # 혹시 몰라 다시하는 복습 : 위에서 x와 y보고 몇행 몇열인지 바로 파악ㄱㄱ
x = np.array( [range(100), range(301,401), range(1, 101)] )
#y = np.array( [range(711, 811), range(101, 201)] )
y = np.array( [range(701, 801)] )
print(x.shape, y.shape)     #(3, 100) (2, 100)

x = np.transpose(x)
y = np.transpose(y)
print(x.shape, y.shape)     #(100, 3) (100, 2)
#x = x.reshape(1, 10, 10, 3)
print(x.shape, y.shape)     #(1, 10, 10, 3) (100, 2)
# 1행무시 / 이미지 데이터는 4차원으로 빠짐
# 제일 앞 행 1은 무시하고 다음 shape을 명시해주면 됨
# none자리를뺀 나머지를 명시해 준다.
# 증권사 데이터도 받아오면 1행 none빼주고 10,10,3 이런식으로 넣어 주게된다
# 헷갈리면 input shape를 하게되면 1행을 none이란 행으로 치고 2열부터
#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

model = Sequential()                # 행의 개수는 무시 가능하지만 x와 y로
# model.add(Dense(10, input_dim=3))   # (100, 3) → (N, 3) : input,output의 feature/열/특성/속성 개수만 맞으면 됨
model.add(Dense(10, input_shape=(3,)))
model.add(Dense(9))
model.add(Dense(8))                # (y컬럼의 개수만 맞고 x컬럼 다를수 있다 ex)5갠데 1개일 수 있음)
model.add(Dense(2))                 # output : y 칼럼개수
model.summary()
