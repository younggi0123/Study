# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】

# Sequential vs model

# File 5. Wine Data

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np



# 1. 데이터
# Data load
datasets = load_wine()
x = datasets.data
y = datasets.target
# print(np.unique(y))             # [0,1,2]

y = to_categorical(y)
# print(y.shape)                  # (178, 3)

# 최소값 최대값
# print(np.min(x), np.max(x))     #  0.13 1680.0

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, random_state=66)

# print(x.shape)                    # (178, 13)
# print(y.shape)                  # (178, 3)
# print(np.unique(y))             # 값을 찍어서 어떤 분류확인 : [0 1 2]이다. 다중분류니까 Softmax & categorical_Crossentropyㄱㄱ

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
# # Scaler fit & transform
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)           # train은 minmaxscaler가 됨
# x_test = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                    # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)

# 2. 모델링

# Sequential Ver Set
model = Sequential()
model.add(Dense(15, input_dim=13))
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='linear'))
model.add(Dense(3, activation='softmax'))               # ★ activation은 softmax ★ activation은 softmax ★ activation은 softmax ★


# Model Ver Set
input1 = Input( shape=(13, ) )                        # Sequential과 동일한 구조로 만들겠다
dense1 = Dense(15)(input1)                          # input1 레이어로부터 받아들였다
dense2 = Dense(9, activation='relu')(dense1)
dense3 = Dense(8, activation='relu')(dense2)
dense4 = Dense(7, activation='relu')(dense3)
dense5 = Dense(6, activation='relu')(dense4)
dense6 = Dense(5, activation='linear')(dense5)
output1 = Dense(3, activation='softmax')(dense6)
model = Model( inputs=input1, outputs=output1 )










# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)

model.fit(x_train, y_train, epochs=100, batch_size=3, verbose=1,
          validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★

# results = model.predict(x_test[:7])
# print(y_test[:7])
# print(results)



#                                                                  【결과 report】                                                                  #
#                                        고정값 : 0.8trainsize, 66th_randomstate, 20patience, 100epochs, 3batch, 0.2val

# model.add(Dense(15, activation='linear', input_dim=13))
# model.add(Dense(9, activation='linear'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(7, activation='linear'))
# model.add(Dense(6))
# model.add(Dense(5, activation='linear'))
# model.add(Dense(3, activation='softmax'))

# 0. Normal그냥ㄱ (Scaler 등 주석으로 지우고 ㄱ)                         0. Normal_Activation='relu'
# loss :  0.7030265927314758                                            loss :  0.285094678401947
# accuracy :  0.75                                                      accuracy :  0.8888888955116272

# a. MinMaxScaler                                                      a. MinMaxScaler_Activation='relu'
# loss :  0.12827223539352417                                           loss :  0.5626896619796753
# accuracy :  0.9722222089767456                                        accuracy :  0.9444444179534912

# b. StandardScaler                                                     b. StandardScaler_Activation='relu'
# loss :  0.32985591888427734                                           loss :  0.007808893918991089
# accuracy :  0.9722222089767456                                        accuracy :  1.0

# c. RobustScaler                                                       c. RobustScaler_Activation='relu'
# loss :  0.196456640958786                                             loss :  0.3210925757884979
# accuracy :  0.9722222089767456                                        accuracy :  0.9722222089767456

# d. MaxAbsScaler                                                       d. MaxAbsScaler_Activation='relu'
# loss :  0.031544264405965805                                          loss :  0.6025871634483337
# accuracy :  1.0                                                       accuracy :  0.6111111044883728
