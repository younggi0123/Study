from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. Data
x = np.array(range(100))    # x=1 bias=1
y = np.array(range(1, 101))
#x가 100이면 y도 100이거나 101이겠지??
# 랜덤 추출(x와 y가 랜덤이 달라지면 안 될 것!!!)

# 나의 시도
#np.random.shuffle(x)
#np.random.shuffle(y)
#np.random.permutation(x)
#np.random.permutation(y)

#train:test=7:3 이렇게 짜르면 x와 y의순서가 다 바뀌어서 못 써
# x와 y의 매칭이 안되고 훈련의 범위가 달라진다면? 차이가 발생한다.
#x_train = x[0:70]
#x_test  = x[71:101]
#y_train = y[0:70]
#y_test  = y[71:101]

# 선생님의 지도
#train과 test를 찾아 주겠다..!
#랜덤값은 고정되어야 객관적 지표로 똑같이 나온다.(랜덤'난수')
#랜덤을 섞는 것도 ☆난수표☆의 기준이 있음 #연산의 parameter

# 지엽적으로 자르는게 아닌, 데이터 전체에서 일정한 비율로 자르는게 좋다
# 일단 API를 불러와서 자르기로 한다
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66)

# train_szie or test_size로 선택해서 가능


#x_train, x_test, y_train, y_test = train_test_split(x,y,
#        train_size=0.7, shuffle=True)

#x와 y가 원래 1씩 차이 나는게 맞아=why?
# random_state 붙인 경우 (고정)
#print(x_test)       #[ 8 93  4  5 52 41  0 73 88 68]
#print(y_test)       #[ 9 94  5  6 53 42  1 74 89 69]

# random_state 제거했을 경우 (결과 계속 바뀜)
#print(x_test) #[88 50 48 97 27 49 89 44  9 45]
#print(y_test) #[89 51 49 98 28 50 90 45 10 46]


#2. modeling
model=Sequential()
model.add(Dense(12,input_dim=1))
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(6))
model.add(Dense(1))

#3. compile&fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

#4. predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([100])
print('100의 예측값 : ', result)
# 3.에 변수_train 4.에 변수_test 들어가는건 기본 (이름차이)

#loss :  5.122918408928534e-11
#100의 예측값 :  [[102.00002]]

#loss :  7.296912407639411e-09
#100의 예측값 :  [[102.00017]]


#loss :  1.6830030062919832e-06
#100의 예측값 :  [[101.99885]]

#loss :  0.002904700580984354
#100의 예측값 :  [[100.88881]]

#loss :  0.0013748232740908861
#100의 예측값 :  [[101.020515]]

#loss :  1.0044368536910042e-05
#100의 예측값 :  [[100.994934]]

#loss :  0.0020302433986216784
#11의 예측값 :  [[100.97145]]

#loss :  4.480817210605892e-07
#100의 예측값 :  [[101.00144]]

#loss :  2.465291970565886e-07
#100의 예측값 :  [[101.00016]]

#loss :  4.8961279389914125e-05
#100의 예측값 :  [[100.984886]]