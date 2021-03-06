# [실습 / 과제]
# cifar100으로 전이학습 모델구성 + 69_2의 preprocess 적용


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D  # 1D는 선만 그어. 2D부터 이미지
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# scaler = StandardScaler()

# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

# y_train = to_categorical(y_train)       # y값은 카테고리컬 해줘여쥐

# y_test = to_categorical(y_test)         # test도 카테고리컬해줘야아아아앆!!!!!!!!!!!!!!!!!!!!!!!


# n = x_train.shape[0]# 이미지갯수 50000
# x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
# x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1

# x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

# m = x_test.shape[0]
# x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)
# print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 100)
# print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 100)
# # reshape또한 일종의 전처리이다.

# # print(x_train[0])
# # print('y_train[0]번째 값 : ', y_train[0])

# # import matplotlib.pyplot as plt
# # plt.imshow(x_train[0], 'gray')
# # plt.show()
# # print(np.unique(y_train, return_counts=True))


# # 2. 모델구성
# from tensorflow.keras.applications import VGG19
# vgg16 = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))
# # vgg16.summary()
# # vgg16.trainable = False # 가중치를 동결시킨다. !          # only VGG16 False 

# model = Sequential()
# model.add(vgg16)
# model.add(Flatten())
# model.add(Dense(400))
# model.add(Dense(300))
# model.add(Dense(200))
# model.add(Dense(150))
# model.add(Dense(100))


# # 3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)

# # Fit
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
#                    verbose=1, restore_best_weights=False)    

# model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
#           validation_split=0.3)  #, callbacks=[es])

# # 4. 평가, 예측
# # Evaluate
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
# print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★

# loss :  7.091963291168213
# accuracy :  0.009999999776482582



########################################################################################
# preprocessing적용 ver.


from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


scaler = StandardScaler()

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

y_train = to_categorical(y_train)       # y값은 카테고리컬 해줘여쥐
y_test = to_categorical(y_test)         # test도 카테고리컬해줘야아아아앆!!!!!!!!!!!!!!!!!!!!!!!

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1

x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)
print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 100)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 100)
# reshape또한 일종의 전처리이다.

# print(x_train[0])
# print('y_train[0]번째 값 : ', y_train[0])

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')
# plt.show()
# print(np.unique(y_train, return_counts=True))


# 2. 모델구성
from tensorflow.keras.applications import VGG19, ResNet101
vgg16 = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))
# vgg16.summary()
# vgg16.trainable = False # 가중치를 동결시킨다. !          # only VGG16 False 

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
          validation_split=0.3)  #, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★

# loss :  nan
# accuracy :  0.009999999776482582