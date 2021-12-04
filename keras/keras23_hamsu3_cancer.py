# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】

# Sequential vs model

# ※ 전처리 및 소스에 대한 설명은 File 1. 참고 ※


# File 3. Breast Cancer Data

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np

# 1. 데이터
# Data load
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x))     #  0.0 4254.0

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66          )

# print(x.shape)                  # (569, 30)
# print(y.shape)                  # (569, )
# print(y_test[:11])              # 값찍방법 1. 값을 찍어보니 [1 1 1 1 1 0 0 1 1 1 0]로 이진분류이다. 시그모이드 ㄱㄱ
# print(np.unique(y))             # 값찍방법 2. [0 1]

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
x_test = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                    # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)

# 2. 모델링

# Sequential Ver Set
model = Sequential()
model.add(Dense(20, input_dim=30))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))               # ★ 마지막은 반드시 시그모이드 ★ 마지막은 반드시 시그모이드 ★ 마지막은 반드시 시그모이드 ★


# Model Ver Set
input1 = Input( shape=(30, ) )                        # Sequential과 동일한 구조로 만들겠다
dense1 = Dense(20)(input1)                          # input1 레이어로부터 받아들였다
dense2 = Dense(15, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(7, activation='linear')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
dense6 = Dense(11, activation='relu')(dense5)
dense7 = Dense(7, activation='relu')(dense6)
output1 = Dense(1, activation='sigmoid')(dense7)
model = Model( inputs=input1, outputs=output1 )

# 모델 세이브
model.save("./_save/keras23.3_save_model.h5")



# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )               # 이진분류 인지 - binary_crossentropy

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto', verbose=1, restore_best_weights=True)    

model.fit(x_train, y_train, epochs=100, batch_size=3, verbose=1, validation_split=0.2, callbacks=[es])



# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# results = model.predict(x_test[:11])
# print(y_test[:11])
# print(results)



#                                                                  【결과 report】                                                                  #
#                                        고정값 : 0.7trainsize, 66th_randomstate, 20patience, 100epochs, 3batch, 0.2val
# model.add(Dense(20, input_dim=30))
# model.add(Dense(15))
# model.add(Dense(10))
# model.add(Dense(7))
# model.add(Dense(10))
# model.add(Dense(11))
# model.add(Dense(7))
# model.add(Dense(1))

# 0. Normal그냥ㄱ (Scaler 등 주석으로 지우고 ㄱ)                         0. Normal_Activation='relu'
# loss: 0.2800 - accuracy: 0.9123                                       loss: 0.5920 - accuracy: 0.6433
# loss :  [0.2800430953502655, 0.9122806787490845]                      [0.5919955968856812, 0.6432748436927795]

# a. MinMaxScaler                                                      a. MinMaxScaler_Activation='relu'
# loss: 0.0636 - accuracy: 0.9766                                       loss: 0.1046 - accuracy: 0.9825
# loss :  [0.06356785446405411, 0.9766082167625427]                     [0.10458622872829437, 0.9824561476707458]


# b. StandardScaler                                                     b. StandardScaler_Activation='relu'
# loss: 0.0925 - accuracy: 0.9708                                       loss: 0.0535 - accuracy: 0.9825
# loss :  [0.0924794152379036, 0.9707602262496948]                      [0.05347554758191109, 0.9824561476707458]

# c. RobustScaler                                                       c. RobustScaler_Activation='relu'
# loss: 0.1119 - accuracy: 0.9708                                       loss: 0.0660 - accuracy: 0.9825
# loss :  [0.11194644123315811, 0.9707602262496948]                     [0.06596861034631729, 0.9824561476707458]

# d. MaxAbsScaler                                                       d. MaxAbsScaler_Activation='relu'
# loss: 0.0897 - accuracy: 0.9766                                       loss: 0.1809 - accuracy: 0.9825
# loss :  [0.08967849612236023, 0.9766082167625427]                     [0.18092066049575806, 0.9824561476707458]