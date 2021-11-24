# 강의 내용 : keras05_mlp3 와 x값만 달라진 모델

import numpy as np
from numpy.core.fromnumeric import transpose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. Data
x = np.array( [range(10)] )
print(x)

x = np.transpose(x)
print(x.shape)             #(10.3)

y = np.array( [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 1.1, 1.2, 1.3, 1.4, 1.5,
                1.6, 1.5, 1.4, 1.3],
                [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
              ]
            )
y = np.transpose(y)
print(y.shape)

#2. 모델 구성
model = Sequential()                
model.add(Dense(5, input_dim=1)) #인풋
model.add(Dense(2))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))               # 아웃풋(개수 유의)


#3. 컴파일
model.compile(loss='mse', optimizer="adam") # 옵티마이저?
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict( [ [9] ] )      # 대괄호 개수 유의
print(' 예측값 : ', result)

#data에 따라서 parameter를 다르게 잡아야(히든레이어 조정하란 말)
