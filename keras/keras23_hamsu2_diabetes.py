# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】

# Sequential vs model

# ※ 전처리 및 소스에 대한 설명은 File 1. 참고 ※


# File 2. Diabetes Data

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np

# 1. 데이터
# Data load
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x))     #  -0.137767225690012 0.198787989657293

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=49          )

# print(x.shape)                  #(442, 10)
# print(y.shape)                  #(442, )

# API 땡겨올때는 어떤 전처리를 사용할 것인지 정의를 해줘야한다.
# 전처리 4대장
# a. MinMaxScaler
# scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
# d. MaxAbsScaler
# scaler = MaxAbsScaler()
# Scaler fit & transform
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test  = scaler.transform(x_test)

# 2. 모델링 비교
# Sequential Ver Set
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(15, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))


# Model Ver Set
input1 = Input( shape=(10, ) )                        # Sequential과 동일한 구조로 만들겠다
dense1 = Dense(16)(input1)                          # input1 레이어로부터 받아들였다
dense2 = Dense(15, activation='relu')(dense1)
dense3 = Dense(14, activation='relu')(dense2)
dense4 = Dense(13, activation='relu')(dense3)
dense5 = Dense(12, activation='relu')(dense4)
dense6 = Dense(11, activation='relu')(dense5)
dense7 = Dense(10, activation='linear')(dense6)
dense8 = Dense(9, activation='relu')(dense7)
dense9 = Dense(8, activation='relu')(dense8)
dense10 = Dense(7, activation='relu')(dense9)
dense11 = Dense(6, activation='relu')(dense10)
dense12 = Dense(5, activation='relu')(dense11)
dense13 = Dense(4, activation='relu')(dense12)
dense14 = Dense(3, activation='relu')(dense13)
dense15 = Dense(2, activation='relu')(dense14)
output1 = Dense(1)(dense15)
model = Model( inputs=input1, outputs=output1 )

model.save("./_save/keras23.2_save_model.h5")

# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=10, validation_split=0.2)


# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)
# Predict
r2 = r2_score(y_test, y_pred)
print('r2스코어 : ', r2)






#               【결과 report】               #
# 고정값 : 0.8trainsize, 49th_randomstate, 30epochs, 10batch, 0.2val
# model.add(Dense(16, input_dim=10))
# model.add(Dense(15))
# model.add(Dense(14))
# model.add(Dense(13))
# model.add(Dense(12))
# model.add(Dense(11))
# model.add(Dense(10))
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(6))
# model.add(Dense(5))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(2))
# model.add(Dense(1))

#                                                                  【결과 report】                                                                  #
#                                        고정값 : 0.8trainsize, 49th_randomstate, 30epochs, 10batch, 0.2val
#                                        
# 0. Normal그냥ㄱ (Scaler 등 주석으로 지우고 ㄱ)                         0. Normal_Activation='relu'
# loss :  1996.262451171875                                             loss :  28301.80859375
# r2스코어 :  0.6253480541219611                                        r2스코어 :  -4.31158994271296

# loss :  1990.76416015625
# r2스코어 :  0.6263799743192804

# a. MinMaxScaler                                                      a. MinMaxScaler_Activation='relu'
# loss :  2065.322021484375                                             loss :  28301.79296875
# r2스코어 :  0.612387188858195                                         r2스코어 :  -4.311586646614474

# b. StandardScaler                                                     b. StandardScaler_Activation='relu'
# loss :  2066.794677734375                                             loss :  27700.884765625
# r2스코어 :  0.6121108327306786                                        r2스코어 :  -4.198810374007589

# c. RobustScaler                                                       c. RobustScaler_Activation='relu'
# loss :  2146.063232421875                                             loss :  28301.826171875
# r2스코어 :  0.5972339506147206                                        r2스코어 :  -4.311593391409948

# d. MaxAbsScaler                                                     d. MaxAbsScaler_Activation='relu'
# loss :  2207.411865234375                                             loss :  28301.662109375
# r2스코어 :  0.5857202518326969                                         r2스코어 :  -4.311562387097087