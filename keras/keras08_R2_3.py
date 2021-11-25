## R2의 이해
# R2의 최고치인 0.1을 찾아보기 !
#R2 Score, R제곱, 설명력, 결정계수 등 일반적으로 R2, 또는 결정계수로 많이 부른다.
#RMSE와 반대로 높을수록 좋은 지표이다. max는 1이다. 0~1 사이의 수치가 값으로 전달된다.

# 직관적으로 말하자면, '설명 가능한 분산(또는 편차)',실제 값의 분산 대비 예측값의 분산 비율
# R2 score는 회귀 모델이 얼마나 '설명력'이 있느냐를 의미. SSR/SST, 실제 값의 분산 대비 예측값의 분산 비율
# 예측 모델과 실제 모델이 얼마나 강한 상관관계를 가지는가?


# R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict) 
# print("R2 : ", r2_y_predict)
#(출처: https://soccerda.tistory.com/131 [soccerda의 IT이야기])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.9, shuffle=True, random_state=1) #random_state낮을수록,trainsize 높을수록.?
#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=1))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(25))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(34))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))
                                                #random_state1로주고, train_size0.9로 주고나서 ↓
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=5000, batch_size=100)     #epo가 낮아지니 정확도 급격히 떨어지는 바람에, epoch높히면서 batch도 늘려 주었다.

#4. 평가, 예측
loss = model.evaluate(x, y)   # evaluate 보여주기 부분
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score   #( mean_squred)
r2 = r2_score(y, y_predict) #ypredict test비교
print('r2스코어 : ', r2)

# loss :  10.006659507751465
# r2스코어 :  0.2964067224672845



#plt.scatter(x,y)        # 점만찍는 것
# plt.plot(x, y_predict, color='red')  # 연속된 선을 긋는 것
# plt.show()

'''
model.add(Dense(30, input_dim=1))
model.add(Dense(13))
model.add(Dense(21))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(4))
model.add(Dense(1))
loss :  0.3801291882991791
r2스코어 :  0.8099354090850042

model.add(Dense(60, input_dim=1))
model.add(Dense(20))
model.add(Dense(65))
model.add(Dense(25))
model.add(Dense(45))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(1))
loss :  0.3800585865974426
r2스코어 :  0.809970715864084




x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.9, shuffle=True, random_state=1) #random_state낮을수록,trainsize 높을수록.
model = Sequential()
model.add(Dense(40, input_dim=1))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(25))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(34))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=200, batch_size=5)
loss :  0.3800000548362732
r2스코어 :  0.8099999761993658




loss :  0.37999996542930603
r2스코어 :  0.809999999999917

epochs=2000, batch_size=100
loss :  0.3799998164176941
r2스코어 :  0.8100000810622134

epochs=3000, batch_size=300
loss :  0.3799998164176941
r2스코어 :  0.810000085830454

epochs=5000, batch_size=100
loss :  0.37999990582466125
r2스코어 :  0.8100000381467055
'''