# 세이브 한 뒤에
# 세이브 한 소스는 w주석처리
# 로드에서 처리
# 데이터수 8005개

import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.layers.pooling import MaxPooling2D

# #1. 데이터
# train_datagen = ImageDataGenerator(
#     rescale = 1./255,
#     horizontal_flip = True,
#     vertical_flip =True,
#     width_shift_range = 0.1,
#     height_shift_range = 0.1,
#     rotation_range = 5,
#     zoom_range = 1.2,
#     shear_range = 0.7,
#     fill_mode = 'nearest'
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255
# )



# xy_train = train_datagen.flow_from_directory(
#     '../_data/image/cat_dog/training_set/',
#     target_size = (100, 100),
#     batch_size = 200,
#     class_mode = 'binary',
#     shuffle = True
# )

# xy_test = test_datagen.flow_from_directory(
#     '../_data/image/cat_dog/test_set/',
#     target_size = (100, 100),
#     batch_size = 200,
#     class_mode = 'binary',
# )

# xy_train[0][0]
# xy_train[0][1]

# # print(xy_train[0][0].shape, xy_train[0][1].shape)   # (160, 150, 150, 3) (160,)
# # print(xy_test[0][0].shape, xy_test[0][1].shape)     # (120, 150, 150, 3) (120,)


# np.save('./_save_npy/keras48_1_train_x.npy', arr=xy_train[0][0])
# np.save('./_save_npy/keras48_1_train_y.npy', arr=xy_train[0][1])
# np.save('./_save_npy/keras48_1_test_x.npy', arr=xy_test[0][0])
# np.save('./_save_npy/keras48_1_test_y.npy', arr=xy_test[0][1])


# 넘파이 세이브 』
####################################################################################################################################




####################################################################################################################################
#『 넘파이 로드 부분


x_train = np.load('./_save_npy/keras48_1_train_x.npy')
y_train= np.load('./_save_npy/keras48_1_train_y.npy')
x_test = np.load('./_save_npy/keras48_1_test_x.npy')
y_test = np.load('./_save_npy/keras48_1_test_y.npy')


# 2. 모델 구성



# model.evaluate에 batch를 명시하지 않아왔지만 원래 batch_size가 존재했단 소리지.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
model = Sequential()
model.add(Conv2D( 32, (2,2), input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=2, strides=1, padding="VALID"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # bc가 낮은거 metrics높은거 잡아주겠지??????
# model.fit(xy_train[0][0], xy_train[0][1])
hist = model.fit(x_train, y_train, epochs=200, batch_size= 32, validation_split = 0.3)


# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss & accuracy 값
# loss :  0.6931984424591064
# accuracy :  0.4699999988079071