# 세이브 한 뒤에
# 세이브 한 소스는 w주석처리
# 로드에서 처리
# 데이터수 8005개

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.preprocessing import image as keras_image

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
# )#Found 8005 images belonging to 2 classes.

# xy_test = test_datagen.flow_from_directory(
#     '../_data/image/cat_dog/test_set/',
#     target_size = (100, 100),
#     batch_size = 200,
#     class_mode = 'binary',
# )#Found 2023 images belonging to 2 classes.

# xy_train[0][0]
# xy_train[0][1]

# print(xy_train[0][0].shape, xy_train[0][1].shape)   # (200, 100, 100, 3) (200,)
# print(xy_test[0][0].shape, xy_test[0][1].shape)     # (200, 100, 100, 3) (200,)


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
model.add(Conv2D( 200, (3,3), input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=2, strides=1, padding="VALID"))
model.add(Dropout(0.3))
model.add(Conv2D( 200, (2,2)))
model.add(MaxPooling2D(pool_size=2, strides=1, padding="VALID"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # bc가 낮은거 metrics높은거 잡아주겠지??????
# model.fit(xy_train[0][0], xy_train[0][1])

import os
path = "./_save/cat_dog_npy_1.h5"
print(os.getcwd())
if os.path.exists(path):
    model.load_weights(path)
else:
    import time
    start = time.time()

    hist = model.fit(x_train, y_train, epochs=200, batch_size= 32, validation_split = 0.3)

    end = time.time()- start
    print("걸린시간 : ", round(end, 3), '초')
    model.save("./_save/cat_dog_npy_1.h5")
    
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
sample_directory = '../_data/image/_predict/cat_dog/'
sample_image = sample_directory + "younggi.jpg"

# 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

# 샘플케이스 평가
loss, acc = model.evaluate(x_test, y_test)    # steps=5
print("Between cat_and_dog data Accuracy : ",str(np.round(acc ,2)*100)+ "%")# 여기서 accuracy는 이 밑의 샘플데이터에 대한 관측치가 아니고 모델 내에서 학습하고 평가한 정확도임
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
# print(classes)          # 

y_predict=model.predict(images, batch_size=40)


print("\n 예측결과")
if(y_predict>=0.5):
    person= y_predict*100
    print(np.round( person, 2), "%의 확률로")
    print(" → '개'입니다. " )
elif(y_predict<0.5):
    horse= (100-(y_predict*100))
    print(np.round( horse, 2), "%의 확률로")
    print(" → '고양이'입니다. ")
else:
    print("ERROR 발생")




# Between cat_and_dog data Accuracy :  45.0%
# 본 샘플데이터s의 정확도는 위와 같음.

#  예측결과
# [[50.18]] %의 확률로
#  → '개'입니다.

# 정답은 없음(사람으로 비교하니까)




# # # 4. 평가, 예측(바꾸기 전)
# # # Evaluate
# # loss = model.evaluate(x_test, y_test)
# # print('loss : ', loss[0])
# # print('accuracy : ', loss[1])

# # # loss & accuracy 값
# # # loss :  0.6931984424591064
# # # accuracy :  0.4699999988079071