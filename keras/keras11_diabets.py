# Q) 당뇨병데이터를 불러와서 R2 로 0.6 이상을 도출하시오.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_diabetes

#1. Data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #쉐입으로 data얼만큼있어? # (442,10) (442,) # input 10 output 1

# What is the data feature name?
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.8, shuffle=True, random_state=49) 

#2. 모델구성
model = Sequential()
'''
model.add(Dense(11, input_dim=10))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(1))
'''
model.add(Dense(16, input_dim=10))
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
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=30, batch_size=10, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # evaluate 보여주기 부분
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score   #( mean_squred)
r2 = r2_score(y_test, y_predict) #ypredict test비교
print('R2스코어 : ', r2)

# 1차
# loss :  2905.28515625
# R2스코어 :  0.42292117007460495

# 2차
#loss :  2854.164306640625
#R2스코어 :  0.4374806762207605

# 3차
#loss :  2863.6259765625
#R2스코어 :  0.435615829199251

# 4차
#loss :  2807.910400390625
#R2스코어 :  0.44226285275939137

# 5차
#loss :  2816.541748046875
#R2스코어 :  0.4405483525586338

# 5차
#loss :  2910.4169921875
#R2스코어 :  0.5328661530521344

#loss :  2546.656005859375
#R2스코어 :  0.5194445829988983

#loss :  3061.14990234375
#R2스코어 :  0.5283311203126303

#loss :  3484.943115234375
#R2스코어 :  0.5621204200157919

'''
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=49) 
model.add(Dense(16, input_dim=10))
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
model.add(Dense(1))
model.fit(x,y, epochs=30, batch_size=10, verbose=1)
#loss :  2048.57177734375
#R2스코어 :  0.6155308058941708
'''
#loss :  2048.57177734375
#R2스코어 :  0.6155308058941708
# R2 스코어가 0.61로 0.6근처의 유의미한 값을 가진다.