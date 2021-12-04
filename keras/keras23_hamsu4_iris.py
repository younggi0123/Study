# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】

# Sequential vs model

# File 4. Iris Data

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np



# 1. 데이터
# Data load
datasets = load_iris()
x = datasets.data
y = datasets.target
y = to_categorical(y)
# print(y.shape)      #(150, 3)

# 최소값 최대값
# print(np.min(x), np.max(x))     #  0.1 7.9

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, random_state=66)

# print(x.shape)                  # (150, 4)
# print(y.shape)                  # (150, )
# print(np.unique(y))             # 값을 찍어서 어떤 분류확인 : [0 1 2]이다. 다중분류니까 Softmax & categorical_Crossentropyㄱㄱ

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
# model = Sequential()
# model.add(Dense(10, activation='relu', input_dim=4))         #input dim 4될 것
# model.add(Dense(9, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(7, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(5, activation='linear'))
# model.add(Dense(3, activation='softmax'))               # ★ activation은 softmax ★ activation은 softmax ★ activation은 softmax ★


# Model Ver Set
input1 = Input( shape=(4, ) )                        # Sequential과 동일한 구조로 만들겠다
dense1 = Dense(10, activation='relu')(input1)                          # input1 레이어로부터 받아들였다
dense2 = Dense(9, activation='relu')(dense1)
dense3 = Dense(8, activation='relu')(dense2)
dense4 = Dense(7, activation='relu')(dense3)
dense5 = Dense(6, activation='relu')(dense4)
dense6 = Dense(5, activation='linear')(dense5)
output1 = Dense(3, activation='softmax')(dense6)
model = Model( inputs=input1, outputs=output1 )




model.save("./_save/keras23.4_save_model.h5")




# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=True)    

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
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
#                                        고정값 : 0.8trainsize, 66th_randomstate, 20patience, 100epochs, 1batch, 0.2val

# model.add(Dense(20, input_dim=30))
# model.add(Dense(10, activation='linear', input_dim=4))
# model.add(Dense(9, activation='linear'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(7, activation='linear'))
# model.add(Dense(6))
# model.add(Dense(5, activation='linear'))
# model.add(Dense(3, activation='softmax'))

# 0. Normal그냥ㄱ (Scaler 등 주석으로 지우고 ㄱ)                         0. Normal_Activation='relu'
# loss :  0.11062383651733398                                           loss :  0.16470149159431458
# accuracy :  0.9333333373069763                                        accuracy :  0.9333333373069763

# a. MinMaxScaler                                                      a. MinMaxScaler_Activation='relu'
# loss :  0.07698750495910645                                           loss :  0.4925209879875183
# accuracy :  0.9666666388511658                                        accuracy :  0.6333333253860474

# b. StandardScaler                                                     b. StandardScaler_Activation='relu'
# loss :  0.06492743641138077                                           loss :  0.2389061003923416
# accuracy :  0.9666666388511658                                        accuracy :  0.9333333373069763

# c. RobustScaler                                                       c. RobustScaler_Activation='relu'
# loss :  0.08720970898866653                                           loss :  0.055108413100242615
# accuracy :  0.9333333373069763                                        accuracy :  0.9666666388511658

# d. MaxAbsScaler                                                       d. MaxAbsScaler_Activation='relu'
# loss :  0.10495732724666595                                           loss :  0.06113617494702339
# accuracy :  0.9666666388511658                                        accuracy :  0.9666666388511658