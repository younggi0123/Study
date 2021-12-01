# 【       Subject : 21'. 12. 01. keras19_1~7까지의 7종류 파일 데이터에 4개지 전처리 방법을 적용해본다.       】

# File 1. Boston Data

################################################################# 【 어떠한 스케일러를 쓸 것인가? 】 #################################################################
# ★정리good-Blog★ (어떠한 스케일러를 쓸 것인가?https://mkjjo.github.io/python/2019/01/10/scaler.html)
'''
데이터를 모델링하기 전에는 반드시 스케일링 과정을 거쳐야 한다.
스케일링을 통해 다차원의 값들을 비교 분석하기 쉽게 만들어주며,
자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지 하고,
독립 변수의 공분산 행렬의 조건수(condition number)를 감소시켜 최적화 과정에서의 안정성 및 수렴 속도를 향상 시킨다.

특히 k-means 등 거리 기반의 모델에서는 스케일링이 매우 중요하다.
'''
################################################################# 【 각각의 Scaler의 특성과 정의 정리 】 #################################################################
'''
1. StandardScaler
평균을 제거하고 데이터를 단위 분산으로 조정한다.
그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다.

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
print(standardScaler.fit(train_data))
train_data_standardScaled = standardScaler.transform(train_data)


2. MinMaxScaler
모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다.
즉, MinMaxScaler 역시 아웃라이어의 존재에 매우 민감하다.

from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler()
print(minMaxScaler.fit(train_data))
train_data_minMaxScaled = minMaxScaler.transform(train_data)


3. MaxAbsScaler
절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다.
양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.

from sklearn.preprocessing import MaxAbsScaler
maxAbsScaler = MaxAbsScaler()
print(maxAbsScaler.fit(train_data))
train_data_maxAbsScaled = maxAbsScaler.transform(train_data)


4. RobustScaler
아웃라이어의 영향을 최소화한 기법이다.
중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있다.

아웃라이어를 포함하는 데이터의 표준화 결과는 아래와 같다.
from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()
print(robustScaler.fit(train_data))
train_data_robustScaled = robustScaler.transform(train_data)

결론적으로 모든 스케일러 처리 전에는 아웃라이어 제거가 선행되어야 한다. 또한 데이터의 분포 특징에 따라 적절한 스케일러를 적용해주는 것이 좋다.


'''
##################################################################################################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# ◎ 전체 데이터를 전처리함       ▶ 전체 전처리는 x  (필기 참고)
#x = x/np.max(x)                # 좌측과 x = x/711.  는 같음
                                # 부동소수점 명시해줌
                                # 하지만 이미지로 작업할 때 이미지 최소값이 256이므로 번잡해서 이걸 쓸때도 있다

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66          )

# print(x.shape)                  #(506, 13)
# print(y.shape)                  #(506, )

# API 땡겨올때는 어떤 전처리를 사용할 것인지 정의를 해줘야한다.
# 전처리 4대장
# a. MinMaxScaler
scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
# d. MaxAbsScaler
# scaler = MaxAbsScaler()
# Scaler fit & transform
scaler.fit(x_train)
x_train = scaler.transform(x_train)           # train은 minmaxscaler가 됨
x_test  = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                                # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)
                                                # fit은 훈련 들어가서 변환 들어가는것
                                                # 변환시키면 변환시킨 결과를 x_train에 넣어줘야..
# 2. 모델링
# Model Set
model = Sequential()
model.add(Dense(13, input_dim=13))
model.add(Dense(8, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

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

# 0. Normal 그냥ㄱ (Scaler 등 주석으로 지우고 ㄱ)                                 0. Normal_Activation:'relu' ⓥ
# loss :  435.7059631347656                                                     loss :  87.35691833496094
# r2스코어 :  -4.273798910412736                                                r2스코어 :  -0.057370934663617756

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
