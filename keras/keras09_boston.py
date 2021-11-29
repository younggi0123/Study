# 강의 내용: 보스톤의 집값 예측 데이터를 이용하여,
# Q) train 0.6~0.8(권장) 기준 R2 0.8 이상을 도출하시오


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)
print(y)

print(x.shape) 
print(y.shape)
#shape를 찍어보니
 # input_dim은 13이겠구나?!! x=컬럼(특성feature열column)이 13개이겠구나!
# (506, 13)     인풋 13
# (506,)        아웃풋 1(벡터 하나니까)

#넘파이 부동소수점 연산
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                test_size=0.3, shuffle=True, random_state=66) #train_size=0.7 대신 test_size=0.3도 가능

#2. 모델구성
model = Sequential()
model.add(Dense(6, input_dim=13))
model.add(Dense(11))
model.add(Dense(20))
model.add(Dense(35))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(65))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

                                                #random_state1로주고, train_size0.9로 주고나서 ↓
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=600, batch_size=13)     #epo가 낮아지니 정확도 급격히 떨어지는 바람에, epoch높히면서 batch도 늘려 주었다.

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # evaluate 보여주기 부분
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score   #( mean_squred)
r2 = r2_score(y_test, y_predict) #ypredict test비교
print('r2스코어 : ', r2)

#loss :  74.22540283203125
#r2스코어 :  0.11497269739041871

#loss :  27.635562896728516
#r2스코어 :  0.6704870960319276

#loss :  29.629220962524414
#r2스코어 :  0.6467157094408784

#loss :  24.215112686157227
#r2스코어 :  0.7112708679975939

#loss :  27.0468807220459
#r2스코어 :  0.6775062767565799

#loss :  24.694961547851562
#r2스코어 :  0.7055494077727813

#loss :  31.627920150756836
#r2스코어 :  0.6171746361946084

#loss :  19.4219970703125
#r2스코어 :  0.7649155280778138

#loss :  24.7364501953125
#r2스코어 :  0.7005892230670283

#loss :  22.159814834594727
#r2스코어 :  0.7317768919474132

#loss :  21.993885040283203
#r2스코어 :  0.7337853320937

# loss :  22.88650131225586
# r2스코어 :  0.7229810629083648

#loss :  18.67340087890625
#r2스코어 :  0.7739765739813558

#loss :  18.740877151489258
#r2스코어 :  0.7731598448196171

#loss :  18.3514347076416
#r2스코어 :  0.777873662920172

#loss :  17.9174747467041
#r2스코어 :  0.783126350750145

#loss :  17.748586654663086
#r2스코어 :  0.7851705556132574

#loss :  17.569915771484375
#r2스코어 :  0.7873331817635036

