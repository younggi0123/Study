# 세이브 한 뒤에
# 세이브 한 소스는 w주석처리
# 로드에서 처리
# 데이터수 8005개

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image

# IDG를 정의한다.
train_datagen = ImageDataGenerator(
    rescale = 1./255,               # 데이터 픽셀 범위 0과 1사이로 scaling하기 위한 인자
    horizontal_flip = True,         # 상하반전(mnist 데이터 등에서 숫자예측시 6과 9는 다른 숫자가 되므로 유의)
    vertical_flip =True,            # 좌우반전
    width_shift_range = 0.1,        # 좌우이동
    height_shift_range = 0.1,       # 상하이동
    rotation_range = 5,             # 회전이동
    zoom_range = 1.2,               # zoom 증폭
    shear_range = 0.7,              # 부동소수점. 층밀리기의 강도입니다. (도 단위의 반시계 방향 층밀리기 각도)
    fill_mode = 'nearest',
    
    validation_split=0.2           # ★ 
)


# ☆★☆★☆★☆★ https://naenjun.tistory.com/17 ☆★☆★☆★☆★
#train-test-valdiation set
train_generator = train_datagen.flow_from_directory(
    '../_data/image/men_women/data',
    target_size = (100, 100),
    batch_size = 79,
    class_mode = 'binary',
    subset='training',
    shuffle = True
) # Found 2520 images belonging to 3 classes.

# D:\_data\image\_predict
validation_generator = train_datagen.flow_from_directory(
    '../_data/image/men_women/data',
    target_size = (100, 100),
    batch_size = 79,
    class_mode = 'binary',
    subset='validation',
    shuffle = True,
)

# print(train_generator[0][0].shape, train_generator[0][1].shape)   # (79, 100, 100, 3) (79,)

np.save('./_save_npy/keras48_4_train_x.npy', arr=train_generator[0][0])
np.save('./_save_npy/keras48_4_train_y.npy', arr=train_generator[0][1])
np.save('./_save_npy/keras48_4_validation_x.npy', arr=validation_generator[0][0])
np.save('./_save_npy/keras48_4_validation_y.npy', arr=validation_generator[0][1])


# # 넘파이 세이브 』



####################################################################################################################################




####################################################################################################################################


#『 넘파이 로드 부분

x_train = np.load('./_save_npy/keras48_4_train_x.npy')
y_train= np.load('./_save_npy/keras48_4_train_y.npy')
x_validation = np.load('./_save_npy/keras48_4_validation_x.npy')
y_validation = np.load('./_save_npy/keras48_4_validation_y.npy')


# 2. 모델
# model.evaluate에 batch를 명시하지 않아왔지만 원래 batch_size가 존재했단 소리지.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
model = Sequential()
model.add(Conv2D( 79, (3,3), input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=2, strides=1, padding="VALID"))
model.add(Conv2D( 48, (2,2)))
model.add(MaxPooling2D(pool_size=2, strides=1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(48, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# print(train_generator.describes)
# print(validation_generator.types)
# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # bc가 낮은거 metrics높은거 잡아주겠지??????
                                                                                       # optimizer='rmsprop'

# steps_per_epoch 함수로 넣는법
# aaa= len(dataset)
# batch=5
# spe=aaa/batch
# steps_per_epoch= spe
import os
path = "./_save/men_women_npy_1.h5"
print(os.getcwd())
if os.path.exists(path):
    model.load_weights(path)
else:
    import time
    start = time.time()

    hist = model.fit(x_train, y_train, epochs=200, batch_size= 32, validation_split = 0.3)

    end = time.time()- start
    print("걸린시간 : ", round(end, 3), '초')
    model.save("./_save/men_women_npy_1.h5")
    
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# import matplotlib.pyplot as plt
# print('loss : ', loss[-1]) # 마지막이 최종 것
# print('val_loss : ', val_loss[-1])
# print('accuracy : ', acc[-1])
# print('val_accuracy : ', val_acc[-1])

# epochs = range(1, len(loss)+1)

# # # #plot 참고 keras13
# plt.figure(figsize=(9, 5))
# plt.plot(epochs, loss, 'r--', label='loss')
# plt.plot(epochs, val_loss, 'r:', label="val_loss")
# plt.plot(epochs, acc, 'b--', label='acc')
# plt.plot(epochs, val_acc, 'b:', label='val_acc')

# plt.grid()
# # plt.title('관측치')
# # plt.ylabel('loss&acc&val_loss&val_acc')
# # plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()

# 샘플 케이스 경로지정
sample_directory = '../_data/image/_predict/men_women/'
sample_image = sample_directory + "younggi.jpg"

# 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

# 샘플케이스 평가
loss, acc = model.evaluate(x_validation, y_validation)    # steps=5
print("Between men and women data Accuracy : ",str(np.round(acc ,2)*100)+ "%")# 여기서 accuracy는 이 밑의 샘플데이터에 대한 관측치가 아니고 모델 내에서 학습하고 평가한 정확도임
print("본 샘플데이터s의 정확도는 위와 같음.")
# print(x_validation, y_validation)

image_ = keras_image.load_img(str(sample_image), target_size=(100, 100))
x = keras_image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /= 255.
# print(x)
images = np.vstack([x])


# argmax는 어차피 classes는 1개뿐이고 값도 하나라 쓸 필요가 없어
# classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS
# print(classes)          # [[0.9999578]]

y_predict = model.predict(images, batch_size=40)
print(y_predict)



# # print("class : ", name)
# y_validation = [range(image_)]
# y_validation= np.array(y_validation)
# # test_set = len(y_validation)  # .next()[1]))
# print("y_validation", y_validation)

# y_validation.reset()
# print(y_validation.class_indices)
# class_indices
# {'paper': 0, 'rock': 1, 'scissors': 2}

print("\n 측정결과>")
if(y_predict>=0.5):
    person= y_predict*100
    print(np.round( person, 2), "%의 확률로")
    print(" → '여성'입니다. " )
elif(y_predict<0.5):
    horse= (100-(y_predict*100))
    print(np.round( horse, 2), "%의 확률로")
    print(" → '남성'입니다. ")
else:
    print("ERROR 발생")
    
# Between horse_or_human data Accuracy :  49.0%
# 본 샘플데이터s의 정확도는 위와 같음.

#  측정결과>
# [[100.]] %의 확률로
#  → '여성'입니다.

# 오답.
