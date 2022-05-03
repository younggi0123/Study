# 【       Subject : 22'. 02. 07. ReduceLR을 적용+ GlobalAveragePooling 써보기       】
# dropout- boston 기반

# File 1. Boston Data

##################################################################################################################################################################
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np
import time

# 1. 데이터
# Data load
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)



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


# 2. 모델링
model = Sequential()                    #MCP도 model은 정의해줘야..
model.add(Dense(13, input_dim=13))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.2))     #20프로를 랜덤하게 빼버린다
model.add(Dense(9, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))
# model.summary()


from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)


# 3. 컴파일,훈련
# cp에 모델과 weight가 들어있으면 로드모델했을때 다 포함되어 있으니 컴파일도 필요없을 것이다
model.compile(loss='mse', optimizer=optimizer) 

es = EarlyStopping( monitor='val_loss', patience=10, mode='min' )

from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5) # patience번동안에 갱신이 안 되면 learning rate 를 50% 감소시키겠다 ( 곱하기 0.5 개념)


                                                                                      
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=2, callbacks=[es, reduce_lr])
end = time.time() - start

# # 4. 평가, 예측
# print("======================= 1. 기본출력 =======================")
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)
r2 = r2_score(y_test, result)



print('lerning rate : ', learning_rate)
print('loss : ', round(loss,4))
print('r2 : ', round(r2,4))
print("걸린시간 : ", round(end))



# 기존) loss :  55.588287353515625 r2 스코어 :  0.3271576401020574


# Learning rate (reduce_lr)
# lerning rate :  0.01
# loss :  23.4472
# r2 :  0.7162
# 걸린시간 :  16


# lerning rate :  0.001
# loss :  23.7716
# r2 :  0.7123
# 걸린시간 :  15