# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】
# Sequential vs model

# 하고 캐글에 제출
# bike_data는 로그변환해야 되는거 알지? 근데 얘는 로그변환하면 rmse가 너무 확 떨어지긴 해

# File 7. Bike_Sharing_Demand Data

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# RSME 정의
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


# 1. 데이터
# Data load
path = "./_data/titanic/"
path = "./_data/bike/"
train = pd.read_csv(path + 'train.csv')
# print(train)        # (10866, 12)
test_file = pd.read_csv(path + 'test.csv')          # test라고 지으면 애매해서
# print(test_file)         # (6493, 9)      train에서 casual,regis,count의 3개가 빠진 데이터
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

# x & y 설정
x = train.drop( ['datetime', 'casual', 'registered', 'count'], axis=1) # 이렇게 4개 빼고 컬럼 구성
test_file = test_file.drop( ['datetime'], axis=1) # model.predict에서 돌아가게 하도록 datetime 오브젝트를 지운다.
y = train['count']

# 로그변환
y = np.log1p(y)

# Train& Test& Val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.7)#, shuffle=True, random_state=66

# Preprocessing
# 전처리 4대장
# a. MinMaxScaler
# scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
# d. MaxAbsScaler
scaler = MaxAbsScaler()
# # Scaler fit & transform
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_test_file = scaler.transform(test_file) # ★실수 잦은 부분★ test.csv도 스케일링해야됨.bike데이터는 평소train안에서만 하는거랑 다르게 따로 파일로 하는거니까.
                                          # +)  모 든  x 는  스 켈 링 되 어 야  한 다 ! ! ! ! !

# 2. 모델링 구성
# Sequential Ver Set
model = Sequential()
model.add(Dense(12, input_dim=8))
model.add(Dense(11, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='linear'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))


# Model Ver Set
input1 = Input( shape=(8, ) )                        # Sequential과 동일한 구조로 만들겠다
dense1 = Dense(12)(input1)                          # input1 레이어로부터 받아들였다
dense2 = Dense(11, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(9, activation='relu')(dense3)
dense5 = Dense(8, activation='relu')(dense4)
dense6 = Dense(7, activation='linear')(dense5)
dense7 = Dense(6, activation='relu')(dense6)
dense8 = Dense(5, activation='relu')(dense7)
dense9 = Dense(4, activation='relu')(dense8)
dense10 = Dense(3, activation='relu')(dense9)
dense11 = Dense(2, activation='relu')(dense10)
output1 = Dense(1)(dense11)
model = Model( inputs=input1, outputs=output1 )





model.save("./_save/keras23.7_save_model.h5")









# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)


# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test)

# Predict
r2 = r2_score(y_test, y_pred)
print('r2스코어 : ', r2)

# RMSLE와 비슷한 효과의 rmse이다.
rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)

############################################제출파일 생성############################################
# results = model.predict(x_test_file)       # x_test_file이 들어감에 유의
# submit_file['count'] = results           # test파일에서 예측한걸 count로 나오면 submit파일에 들어가진다

# print(submit_file[:10])                 # submit_file이라는 변수에 count가 들어갔고 csv로 보내준다.

# submit_file.to_csv(path + "bike_preprocessingTest_submit_ver.csv", index=False)


#

############################################【결과 report】############################################

# 고정값 : 0.3trainsize, 20patience, 10epochs, 32batch, 0.2val
# model.add(Dense(15, activation='linear', input_dim=13))
# model.add(Dense(9, activation='linear'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(7, activation='linear'))
# model.add(Dense(6))
# model.add(Dense(5, activation='linear'))
# model.add(Dense(3, activation='softmax'))

# 0. Normal그냥ㄱ (Scaler 등 주석으로 지우고 ㄱ)                         0. Normal_Activation='relu'
# loss :  1.7997132539749146                                            loss :  4.101627349853516
# r2스코어 :  0.08157018699570995                                       r2스코어 :  -1.018971103829546
# RMSE :  1.341533908118344                                             RMSE :  2.025247464188976

# a. MinMaxScaler                                                      a. MinMaxScaler_Activation='relu'
# loss :  1.7438613176345825                                            loss :  4.1329216957092285
# r2스코어 :  0.14053676621012612                                       r2스코어 :  -1.0568490251268616
# RMSE :  1.3205532784465774                                            RMSE :  2.032958803673275

# b. StandardScaler                                                     b. StandardScaler_Activation='relu'
# loss :  1.4958868026733398                                            loss :  4.077948570251465
# r2스코어 :  0.25647147060121256                                        r2스코어 :  -0.9735041483918412
# RMSE :  1.2230645436493621                                            RMSE :  2.0193928011069393

# c. RobustScaler                                                       c. RobustScaler_Activation='relu'
# loss :  1.5090842247009277                                            loss :  4.049920558929443
# r2스코어 :  0.25004496468421566                                       r2스코어 :  -0.9513021414953273
# RMSE :  1.2284477665803168                                            RMSE :  2.0124415477688644

# d. MaxAbsScaler                                                       d. MaxAbsScaler_Activation='relu'
# loss :  1.6024384498596191                                            loss :  1.4498529434204102
# r2스코어 :  0.19806124095883715                                        r2스코어 :  0.2733605721074287
# RMSE :  1.2658747117930464                                            RMSE :  1.2040983455135978



