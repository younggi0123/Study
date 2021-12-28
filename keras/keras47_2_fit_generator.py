import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.pooling import MaxPooling2D


# generator 완벽정리 ☆ generator 완벽정리 ☆ generator 완벽정리 ☆ generator 완벽정리 ☆
# https://tykimos.github.io/2017/03/08/CNN_Getting_Started/
# generator 완벽정리 ☆ generator 완벽정리 ☆ generator 완벽정리 ☆ generator 완벽정리 ☆
# https://tykimos.github.io/2017/03/08/CNN_Getting_Started/
# generator 완벽정리 ☆ generator 완벽정리 ☆ generator 완벽정리 ☆ generator 완벽정리 ☆
# https://tykimos.github.io/2017/03/08/CNN_Getting_Started/

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip =True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 5,
    zoom_range = 1.2,
    shear_range = 0.7,
    fill_mode = 'nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train/',
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary',
    shuffle = True
)
xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary',
)
# print(xy_train[0][0].shape, xy_train[0][1].shape)     # (5, 150, 150, 3) (5,)


# 2. 모델
# model.evaluate에 batch를 명시하지 않아왔지만 원래 batch_size가 존재했단 소리지.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D( 16, (2,2), input_shape=(150, 150, 3)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # bc가 낮은거 metrics높은거 잡아주겠지??????
# model.fit(xy_train[0][0], xy_train[0][1])
hist = model.fit_generator(xy_train, epochs=200, steps_per_epoch=32,    # batch_size를 명시할 필요가 없는 대신에 epo당 step을 몇 번 할지를 명시 한다.
                                                                       # 전체데이터 나누기 batchsize( Total/Batch = 160/5= 32 ) => ※무조건 써줘야 함※
                                                                       # 전체데이터는 경로 들가서 몇 개인지 ㄱㄱ 여기선 80,80개였음
                           validation_data=xy_test,
                           validation_steps= 4,                         # 검증데이터 / 베치사이즈
                           # verbose = 1
                           )

# 그래프 시각화
# history 시리즈 정리 ㄱㄱ

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

import matplotlib.pyplot as plt
print('loss : ', loss[-1]) # 마지막이 최종 것
print('val_loss : ', val_loss[-1])
print('accuracy : ', acc[-1])
print('val_accuracy : ', val_acc[-1])

epochs = range(1, len(loss)+1)

# #plot keras13
plt.figure(figsize=(9, 5))
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='loss')
# plt.plot(hist.history['acc'], marker='.', c='green', label='acc')
# plt.plot(hist.history['val_acc'], marker='.', c='yellow', label='val_acc')

plt.plot(epochs, loss, 'r--', label='loss')
plt.plot(epochs, val_loss, 'r:', label="val_loss")
plt.plot(epochs, acc, 'b--', label='acc')
plt.plot(epochs, val_acc, 'b:', label='val_acc')

plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()




# accuracy가 아닌 acc로 쳤을때 오류 나는 경우
# => (https://needneo.tistory.com/30)