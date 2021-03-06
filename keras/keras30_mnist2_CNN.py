from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten # 1D는 선만 그어. 2D부터 이미지
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)         # test도 카테고리컬해줘야 10으로 바뀜
print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000, )
print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000, )
#reshape또한 일종의 전처리이다.
x_train = x_train.reshape(60000, 28, 28, 1) # 전체를 다 곱했을때 같으면 상관없다
# x_test = x_test.reshape(60000, 28, 14, 1) # 이것도 가능하다
x_test = x_test.reshape(10000, 28, 28, 1)
                                            # 위치는 바뀌지 않는다
                                            # convolutionlayer로 넣기 위해 4차원으로 바꾼것
print(x_train.shape)                        # (60000,28,28,1)

print(np.unique(y_train, return_counts=True))   #10개 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#input data shape rate : 60000,20 4차원


##### 문제
# train test는 이미 분리되어있으니 train은 valsplit으로 하고 test는 평가용으로만
# 평가지표 acc 0.98 이상으로.


# print(y)
# print(y.shape)
print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,)

# 2. 모델구성
#hint conv-layer는 3~4개
model  =  Sequential() 
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(28, 28, 1 ) ))
model.add(Conv2D(5, (2,2), activation="relu") )
model.add(Dropout(0.2))
model.add(Conv2D(7, (2,2), activation="relu") )
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation='softmax'))




# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

model.fit(x_train, y_train, epochs=16, batch_size=32, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
          validation_split=0.3, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★

# loss :  0.0760224387049675 accuracy :  0.9810000061988831