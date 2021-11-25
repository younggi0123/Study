# 1. R2를 음수가 아닌 0.5 이하로 만들것 (좋은 데이터를 이용해 안 좋은 결과값을 도출하여 어떤점이 안 좋은 케이스인지 각인)
# 2. 데이터는 건들지 않는다
# 3. 레이어는 인풋 아웃풋 포함 6개 이상
# 4. batch_size = 1
# 5. epochs는 100 이상
# 6. 히든레이어의 노드는 10개 이상 1000개 이하
# 7. train 70%


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.7, shuffle=True, random_state=66) # train size 주로 60~80퍼 정도
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(10))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(1000))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(10))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=101, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  10.006659507751465
# r2스코어 :  0.2964067224672845



#plt.scatter(x,y)        # 점만찍는 것
# plt.plot(x, y_predict, color='red')  # 연속된 선을 긋는 것
# plt.show()


#loss :  131.61349487304688
#r2스코어 :  0.849478301515995

