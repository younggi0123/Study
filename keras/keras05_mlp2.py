# 강의 내용 : keras03~04에서 배운 행렬, 전치행렬을 응용한 문제 풀이
# mlp : multi layer perceptron

# Q) 결과값도출
# 20 내외 (input_dim= 3, output= 1)

import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

#1. Data
# x와 y가 전치된 잘못된 데이터가 주어짐 / shape하면 3,10 나올 것이다
x = np.array( [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 1.1, 1.2, 1.3, 1.4, 1.5,
                1.6, 1.5, 1.4, 1.3],
                [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
              ]
            )
y = np.array( [11, 12, 13, 14, 15, 16, 17, 18, 19, 20] )

x2= np.transpose(x)
print(x.shape)
print(y.shape)
print(x2.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(15, input_dim=3)) #x의 벡터 3
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일,
model.compile(loss='mse', optimizer='adam')
model.fit(x2, y, epochs=30, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x2, y)
print('loss : ', loss)

result = model.predict([10, 1.3, 1])
print('예측값 : ', result)

# 컬럼셋이면 가중치3 하지만 가중치는 하나
# 20 +- 0.5가 나와야 함
# feature? column? 


#왜 전치?