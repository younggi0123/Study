# 【       Subject : 21'. 12. 01. keras19_1~7까지의 7종류 파일 데이터에 4개지 전처리 방법을 적용해본다.       】

# File 6. Fetch_covtype Data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np



# 1. 데이터
# Data load
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(np.unique(y))               # 1 2 3 4 5 7 , 7개의 라벨
y = to_categorical(y)

# 최소값 최대값
# print(np.min(x), np.max(x))     #  -173.0 7173.0

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, random_state=66)

# print(x.shape)                    # (581012, 54)
# print(y.shape)                    # (581012, 8)
# print(x_train.shape, y_train.shape) #(464809, 54) (464809, 8)
# print(x_test.shape, y_test.shape)   #(116203, 54) (116203, 8)

# API 땡겨올때는 어떤 전처리를 사용할 것인지 정의를 해줘야한다.
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
x_train = scaler.transform(x_train)           # train은 minmaxscaler가 됨
x_test = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                    # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)

# 2. 모델링
# Model Set
model = Sequential()
model.add(Dense(70, input_dim=54))         #input_dim 54될 것
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(8, activation='softmax'))               # ★ activation은 softmax ★ activation은 softmax ★ activation은 softmax ★

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)

model.fit(x_train, y_train, epochs=20, batch_size=200, verbose=1,
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
#                                        고정값 : 0.8trainsize, 66th_randomstate, 20patience, 20epochs, 200batch, 0.2val

# model.add(Dense(70, activation='linear', input_dim=54))
# model.add(Dense(50, activation='linear'))
# model.add(Dense(30, activation='linear'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(8, activation='softmax')) 

# 0. Normal그냥ㄱ (Scaler 등 주석으로 지우고 ㄱ)                         0. Normal_Activation='relu'
# loss :  0.662109375                                                  loss :  0.5736944675445557
# accuracy :  0.713010847568512                                        accuracy :  0.7505744099617004

# a. MinMaxScaler                                                      a. MinMaxScaler_Activation='relu'
# loss :  0.6391712427139282                                            loss :  0.3999856114387512
# accuracy :  0.7223823666572571                                        accuracy :  0.8321127891540527

# b. StandardScaler                                                     b. StandardScaler_Activation='relu'
# loss :  0.6348841190338135                                            loss :  0.3706323206424713
# accuracy :  0.7250845432281494                                        accuracy :  0.8475770950317383

# c. RobustScaler                                                       c. RobustScaler_Activation='relu'
# loss :  0.6328914165496826                                            loss :  0.3553144931793213
# accuracy :  0.7219434976577759                                        accuracy :  0.8521897196769714

# d. MaxAbsScaler                                                       d. MaxAbsScaler_Activation='relu'
# loss :  0.6386694312095642                                            loss :  0.41966578364372253
# accuracy :  0.7188368439674377                                        accuracy :  0.824083685874939


