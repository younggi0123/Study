from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터 
x = np.array([1.2,3])
y = np.array([1.2,3])



# 2. 모델구성
# 히든레이어의 크기는 어떻게 정하나여?(https://www.clien.net/service/board/kin/10588915)
#(https://data-newbie.tistory.com/140)
model = Sequential()
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) #output은 고정이므로 건들 수 없다

# 인풋 아웃풋을 제외한 중간단을 조정하는 것을 hyper parameter tuning이라고 함

#########################################################################


# 3. 컴파일,  훈련
# 에포치값 30고정
model.compile(loss='mse', optimizer='adam')
# batch와 epochs를 조절하는 것도 hyper parameter tuning이라고 함
model.fit(x, y, epochs=30, batch_size=1)



# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)



# 결국 4가 나와야 한다


'''
# 1차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
loss :  2.7980135200778022e-05
4의 예측값 :  [[3.9941764]]

# 2차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))
loss :  0.05135275050997734
4의 예측값 :  [[3.9217663]]

# 3차
model.add(Dense(7, input_dim=1)) #input
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
loss :  1.4550490379333496
4의 예측값 :  [[1.7124254]]

# 4차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(3))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
loss :  0.687178909778595
4의 예측값 :  [[2.2880802]]

# 5차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(2))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))
loss :  0.8924015164375305
4의 예측값 :  [[2.0700173]]

# 6차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(2))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))
loss :  2.2967112064361572
4의 예측값 :  [[0.959894]]

# 7차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))
loss :  0.07774841040372849
4의 예측값 :  [[4.0662327]]

# 8차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(1))
loss :  0.015586016699671745
4의 예측값 :  [[3.718718]]

# 9차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(9))
model.add(Dense(3))
model.add(Dense(1))
loss :  0.018356041982769966
4의 예측값 :  [[3.7051182]]

# 10차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(11))
model.add(Dense(3))
model.add(Dense(1))
loss :  0.12977886199951172
4의 예측값 :  [[3.5951712]]

# 11차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))
loss :  0.005627483129501343
4의 예측값 :  [[3.9625888]]

# 12차
model.add(Dense(5, input_dim=1)) #input
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
loss :  0.0036743003875017166
4의 예측값 :  [[3.999486]]

########## 12차에서 약 3.999의 유의미한 결과를 보인다. ##########


'''