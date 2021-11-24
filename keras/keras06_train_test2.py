# 강의 내용 : trainData와 testData를 나누는 것은 과적합 방지이다.!
# 나누지 않고 하면 이미 답을 알고 있기에 (앞에 까지 내용은 가중치가 이미 적합된 상태임) 문제가 생기니까 데이터를 짤라서 진행하는 것이다.

# 과제
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. Data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# train과 test비율을 8:2으로 분리하시오.( ★ 구간 나누어 리스트에 슬라이싱으로 커팅)
#리스트 슬라이싱 방법(https://www.everdevel.com/Python/list-slicing/)
# x_train = np.array(x[0:7])
# x_test  = np.array(x[8:9])
# y_train = np.array(y[0:7])
# y_test  = np.array(y[8:9]) #np.array를 칠 필요가 없지 !
x_train = x[0:7]
x_test  = x[8:9]
y_train = y[0:7]
y_test  = y[8:9]

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) #train data가 7개이므로 1epo당 7번씩 돌고, x와 y를 비교하므로 전체 훈련은 700번이 될 것
# batch가 2면 일괄작업 사이즈가 2이므로 x&y비교 1&1 1번, 2&2 2번 ...

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)

#1차
#loss :  0.0019276717212051153
#11의 예측값 :  [[10.933484]]

#2차
#loss :  0.34510537981987
#11의 예측값 :  [[10.129826]]

#3차
#loss :  9.454962855670601e-05
#11의 예측값 :  [[10.985251]]

#4차
#loss :  0.00022799566795583814
#11의 예측값 :  [[10.977375]]

#5차
#model.add(Dense(10, input_dim=1))
#model.add(Dense(4))
#model.add(Dense(8))
#model.add(Dense(3))
#model.add(Dense(1))
#model.fit(x_train, y_train, epochs=100, batch_size=1)
#결과값
#loss :  2.518296241760254e-06
#11의 예측값 :  [[10.9977]]
# → 예측값이 10.9977인 유의미한 결과를 나타낸다.
