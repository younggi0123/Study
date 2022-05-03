# 【       Subject : 22'. 02. 07. ReduceLR을 적용+ GlobalAveragePooling 써보기       】
# dropout- diabetes 기반

# File 2. Diabetes Data

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
import time
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
model.add(Dense(64, input_dim=10))
model.add(Dense(32, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(2, activation='linear'))
model.add(Dense(1))




from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(lr=learning_rate)



# 3. 컴파일,훈련
model.compile(loss='mse', optimizer=optimizer) 


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping( monitor='val_loss', patience=10, mode='min' ) # [] 리스트=두개이상 #, restore_best_weights=True


from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

                                                                       
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1, callbacks=[es, reduce_lr])
end = time.time() - start



# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)
r2 = r2_score(y_test, result)





print('lerning rate : ', learning_rate)
print('loss : ', round(loss,4))
print('r2 : ', round(r2,4))
print("걸린시간 : ", round(end))


# 기 존:  loss :  27499.564453125 r2 스코어 :  -4.161027350492502





# 적용 후
# Epoch 00041: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# Epoch 00046: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
# lerning rate :  0.01
# loss :  2024.3934
# r2 :  0.6201