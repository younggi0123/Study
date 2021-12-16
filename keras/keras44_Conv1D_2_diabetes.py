# Q) 당뇨병데이터를 불러와서 R2 로 0.62 이상을 도출하시오.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_diabetes

#1. Data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (442,10) (442,) # input 10 output 1

x = x.reshape(442,10,1)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49) 

#2. 모델구성
model = Sequential()
model.add(Conv1D(16, 2, input_shape=(10,1)))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Flatten())
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
import time
start = time.time()
model.fit(x,y, epochs=30, batch_size=10, verbose=1)
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')




#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # evaluate 보여주기 부분
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score   #( mean_squred)
r2 = r2_score(y_test, y_predict) #ypredict test비교
print('R2스코어 : ', r2)

# 걸린시간 :  21.6 초
# loss :  2110.692138671875
# R2스코어 :  0.6038722550403844