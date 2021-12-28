# kaggle.com/c/dogs-vs-cats/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

from tensorflow.python.keras.layers.core import Dropout

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# D:\_data\image\brain

batch_num = 20

xy_train = train_datagen.flow_from_directory(
    '../_data/image/cat_dog/training_set',
    target_size=(25, 25),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=batch_num,
    class_mode='binary',
    shuffle=True
)       # Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set',
    target_size=(25, 25),
    batch_size=batch_num,
    class_mode='binary'    
)       # Found 2023 images belonging to 2 classes.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001D297074F40>

print(xy_train[31])       # 마지막 batch
print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][0].shape, xy_train[0][1].shape)         # (10, 150, 150, 3), (10,)   # 흑백은 알아서 찾아라

print(type(xy_train))       # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(4,(2,2), padding='same', input_shape = (25,25,3)))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16,activation='relu')) 
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
# model.fit(xy_train[0][0], xy_train[0][1])

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       
model_path = "".join([filepath, 'k48_1_cat_dog_IDG_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)

start = time.time()
hist = model.fit_generator(xy_train, epochs=3, steps_per_epoch=401,    # steps_per_epoch = 전체 데이터 수 / batch = 160 / 5 = 32
                    validation_data=xy_test,
                    validation_steps=4, callbacks=[es, mcp]
                    )
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 점심때 그래프 그려보아요

# import matplotlib.pyplot as plt
# # summarize history for accuracy
# plt.plot(acc)
# plt.plot(val_acc)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(loss)
# plt.plot(val_loss)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

# plt.plot(epochs, loss, 'r--', label="loss")
# plt.plot(epochs, val_loss, 'r:', label="loss")
# plt.plot(epochs, acc, 'b--', label="acc")
# plt.plot(epochs, val_acc, 'b:', label="val_acc")

#4. 평가, 예측

# print("-- Evaluate --")
# scores = model.evaluate_generator(xy_test, steps=5)
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# print("-- Predict --")
# output = model.predict_generator(xy_test, steps=5)
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# print(xy_test.class_indices)
# print(output)


# 걸린시간 :  797.231 초
# loss :  0.6461736559867859
# val_loss :  0.632144033908844
# acc :  0.6287320256233215
# val_acc :  0.6499999761581421