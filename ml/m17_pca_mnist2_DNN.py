# 실습

# 아까 4가지로 모델 만들기
# 784개 DNN으로 만든 것(최상의 성능인 것 // 0.978 이상)과 비교!

# time 체크 (fit에서 잴 것)

# < 예상 결과 >

# 나의 최고의 DNN
# time = ????
# acc = ????

# 2. 나의 최고의 CNN 
# time = ????
# acc = ????

# 3. PCA 0.95
# time = ????
# acc = ????

# 4. PCA 0.99
# time = ????
# acc = ????

# 5. PCA 0.999
# time = ????
# acc = ????

# 4. PCA 1.0
# time = ????
# acc = ????


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

# 1.2. 스케일링
# x_train = x_train.reshape(60000, 28*28).astype('float32')/255
# x_test = x_test.reshape(10000, 28*28).astype('float32')/255
scaler = MinMaxScaler()   
x_train = scaler.fit_transform(x_train)     
x_test = scaler.transform(x_test)


# # 1.1.append로 행으로 붙임
# x = np.append( x_train, x_test, axis=0 )
# x = np.reshape(x, (x.shape[0], (x.shape[1]*x.shape[2])) )

# 1.2. PCA
n_components = 154
pca = PCA(n_components)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# pca_EVR = pca.explained_variance_ratio_

# print(pca_EVR)
# print(sum(pca_EVR))

# cumsum = np.cumsum(pca_EVR)
# print(np.argmax(cumsum >= 0.95)+1 )
# print(np.argmax(cumsum >= 0.99)+1 )
# print(np.argmax(cumsum >= 0.999)+1 )
# print(np.argmax(cumsum) +1)



# 2. 모델구성
model  =  Sequential()
model.add(Dense(64 , input_shape=(n_components, )) )
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation='softmax'))



# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)
# Fit
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
#                    verbose=1, restore_best_weights=False)    

# model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
#           validation_split=0.3, callbacks=[es])
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.3 )

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( x_test )
#print("예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict) # y_predict test비교
print('r2스코어 : ', r2)


