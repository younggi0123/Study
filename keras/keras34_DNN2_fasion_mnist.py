from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D  # 1D는 선만 그어. 2D부터 이미지
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 실습 !

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000, 10)
# print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000, 10)
y_train = to_categorical(y_train)       # y값은 카테고리컬 해줘여쥐
y_test = to_categorical(y_test)         # test도 카테고리컬해줘야아아아앆!!!!!!!!!!!!!!!!!!!!!!!

scaler = StandardScaler()

n = x_train.shape[0]
x_train_reshape = x_train.reshape(n,-1)
x_train = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1))

# print(x_train.shape)            # (60000, 784)
# print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000, 10)
# print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000, 10)

# 2. 모델구성
model  =  Sequential()
model.add(Dense(64 , input_shape=(784, )) )
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation='softmax'))



# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
          validation_split=0.3, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( x_test )
#print("예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict) #ypredict test비교
print('r2스코어 : ', r2)