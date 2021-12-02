# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】

# Sequential vs model

# File 1. Boston Data

##################################################################################################################################################################
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np

# 1. 데이터
# Data load
datasets = load_boston()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x))     #  0.0  711.0

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66          )



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
# x_train = scaler.transform(x_train)           # train은 minmaxscaler가 됨
# x_test  = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                                # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)
                                                # fit은 훈련 들어가서 변환 들어가는것
                                                # 변환시키면 변환시킨 결과를 x_train에 넣어줘야..

# 2. 모델링 비교

# Sequential Ver Set
# model = Sequential()
# model.add(Dense(13, input_dim=13))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(9, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(1))

# Model Ver Set
input1 = Input( shape=(13, ) )                        # Sequential과 동일한 구조로 만들겠다
dense1 = Dense(13)(input1)                          # input1 레이어로부터 받아들였다
dense2 = Dense(8, activation='relu')(dense1)
dense3 = Dense(9)(dense2)
dense4 = Dense(6)(dense3)
output1 = Dense(1)(dense4)
model = Model( inputs=input1, outputs=output1 )





# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)
# Predict
r2 = r2_score(y_test, y_pred)
print('r2스코어 : ', r2)






#                                                                  【결과 report】                                                                  #

#                                        고정값 : 0.7trainsize, 66th_randomstate, 10epochs, 32batch, 0.2val
# model.add(Dense(13, input_dim=13))
# model.add(Dense(8, activation='linear'))                                      activation = 'relu'
# model.add(Dense(9, activation='linear'))                                      activation = 'relu'
# model.add(Dense(6, activation='linear'))                                      activation = 'relu'
# model.add(Dense(1))

# 0. Normal 그냥ㄱ (Scaler 등 주석으로 지우고 ㄱ)                                 0. Normal_Activation:'relu' ⓥ                                 0. lyaer : model
# loss :  435.7059631347656                                                     loss :  87.35691833496094                                      loss :  62.85050964355469
# r2스코어 :  -4.273798910412736                                                r2스코어 :  -0.057370934663617756                               r2스코어 :  0.23925541646177217

# loss :  74.9680404663086
# r2스코어 :  0.0925844641714626


# a. MinMaxScaler                                                               a. MinMaxScaler_Activation:'relu'
# loss :  112.34425354003906                                                    loss :  459.6265563964844
# r2스코어 :  -0.35981841620952637                                              r2스코어 :  -4.563334515031043


# b. StandardScaler                                                             b. StandardScaler_Activation:'relu'
# loss :  443.7129821777344                                                     loss :  438.4278259277344
# r2스코어 :  -4.3707162403989495                                               r2스코어 :  -4.3067447895817095


# c. RobustScaler                                                               c. RobustScaler_Activation:'relu'
# loss :  310.0614318847656                                                     loss :  336.99090576171875
# r2스코어 :  -2.7529934564458003                                               r2스코어 :  -3.078948790494019

# d. MaxAbsScaler                                                               # d. MaxAbsScaler_Activation:'relu'
# loss :  266.1025695800781                                                      loss :  182.71656799316406
# r2스코어 :  -2.220913697678784                                                  r2스코어 :  -1.2116070370056753
