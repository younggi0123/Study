# keras 33-cifar 100 기반



from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D  # 1D는 선만 그어. 2D부터 이미지
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

y_train = to_categorical(y_train)       # y값은 카테고리컬 해줘여쥐
y_test = to_categorical(y_test)         # test도 카테고리컬해줘야아아아앆!!!!!!!!!!!!!!!!!!!!!!!
# print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 100)
# print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 100)
# reshape또한 일종의 전처리이다.

# print(x_train[0])
# print('y_train[0]번째 값 : ', y_train[0])

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')
# plt.show()
# print(np.unique(y_train, return_counts=True))

scaler = StandardScaler()

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

y_train = to_categorical(y_train)       # y값은 카테고리컬 해줘여쥐

y_test = to_categorical(y_test)         # test도 카테고리컬해줘야아아아앆!!!!!!!!!!!!!!!!!!!!!!!


n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1

x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)


# 2. 모델구성
#hint conv-layer는 3~4개
model  =  Sequential() 
model.add(Conv2D(300, kernel_size=(4,4), strides=1, padding='same', input_shape=(32, 32, 3) )) 
# model.add(Conv2D(7, kernel_size=(3,3), input_shape=(28, 28, 1 ) ))
model.add(MaxPooling2D()) # 맥스풀링 삽입은 conv2D 다음이다.
model.add(Conv2D(200, (3,3), activation="relu") )
model.add(Dropout(0.3))
model.add(Conv2D(120, (2,2), activation="relu") )
model.add(Dropout(0.3))
model.add(Conv2D(80, (2,2), activation="relu") )
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(100, activation='softmax'))




# 3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer , metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)




# Fit

#################################################################추가된 내용##########################################################################
# EarlyStopping에 러닝레이트 경사하강에도 EarlyStopping 적용( learning rate 가 낮아지다가 오차가 줄어줄며 좌우로 핑퐁하며 성능향상이 없을때 es걸겠다.)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',patience=20, mode='min',  # mode='auto'
                   verbose=1)                                   # , restore_best_weights=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5) # patience번동안에 갱신이 안 되면 learning rate 를 50% 감소시키겠다 ( 곱하기 0.5 개념)
######################################################################################################################################################



    


import time
start = time.time()
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
          validation_split=0.3, callbacks=[es, reduce_lr])          # 정의한 reduce_lr을 callback함수에 위치시켜 준다.
end = time.time()



# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('lerning rate : ', learning_rate)
print('loss : ', round(loss[0],4))
print('accuracy : ', round(loss[1],4))
print("걸린시간 : ", round(end - start,4))



# ReduceLROnPlateau reducing learning가 갱신되는 것을 터미널에서 확인하여 본다 #
# ex. 
# 1093/1094 [============================>.] - ETA: 0s - loss: 1.9029 - accuracy: 0.4849 
# Epoch 00009: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.

# lerning rate :  0.0001
# loss :  5.82
# accuracy :  0.2929
# 걸린시간 :  283.9352


# lerning rate :  0.001
# loss :  5.7236
# accuracy :  0.2887
# 걸린시간 :  297.2653


# lerning rate :  0.01
# loss :  6.2583
# accuracy :  0.282
# 걸린시간 :  1179.9043

