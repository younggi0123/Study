# kaggle.com/c/dogs-vs-cats/data
# 데이터수 2520개
# ★https://necromuralist.github.io/Neurotic-Networking/posts/keras/rock-paper-scissors/★

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image

# # IDG를 정의한다.
# train_datagen = ImageDataGenerator(
#     rescale = 1./255,               # 데이터 픽셀 범위 0과 1사이로 scaling하기 위한 인자
#     horizontal_flip = True,         # 상하반전(mnist 데이터 등에서 숫자예측시 6과 9는 다른 숫자가 되므로 유의)
#     vertical_flip =True,            # 좌우반전
#     width_shift_range = 0.1,        # 좌우이동
#     height_shift_range = 0.1,       # 상하이동
#     rotation_range = 5,             # 회전이동
#     zoom_range = 1.2,               # zoom 증폭
#     shear_range = 0.7,              # 부동소수점. 층밀리기의 강도입니다. (도 단위의 반시계 방향 층밀리기 각도)
#     fill_mode = 'nearest',
    
#     validation_split=0.2           # ★ 
# )

# # ☆★☆★☆★☆★ https://naenjun.tistory.com/17 ☆★☆★☆★☆★
# #train-test-valdiation set
# train_generator = train_datagen.flow_from_directory(
#     '../_data/image/rps/',
#     target_size = (100, 100),
#     batch_size = 40,
#     class_mode = 'categorical', # categorical쓰면 one-hot 안 해도됨
#     subset='training',
#     shuffle = True
# ) # Found 2520 images belonging to 3 classes.

# # D:\_data\image\_predict
# validation_generator = train_datagen.flow_from_directory(
#     '../_data/image/rps/',
#     target_size = (100, 100),
#     batch_size = 40,
#     class_mode = 'categorical',
#     subset='validation',
#     shuffle = True,
# )

# np.save('./_save_npy/keras48_1_train_generator_x.npy', arr=train_generator[0][0])
# np.save('./_save_npy/keras48_1_train_generator_y.npy', arr=train_generator[0][1])
# np.save('./_save_npy/keras48_1_validation_generator_x.npy', arr=validation_generator[0][0])
# np.save('./_save_npy/keras48_1_validation_generator_y.npy', arr=validation_generator[0][1])


# 넘파이 세이브 』
####################################################################################################################################


####################################################################################################################################
#『 넘파이 로드 부분


x_train = np.load('./_save_npy/keras48_1_train_generator_x.npy')
y_train= np.load('./_save_npy/keras48_1_train_generator_y.npy')
x_validation = np.load('./_save_npy/keras48_1_validation_generator_x.npy')
y_validation = np.load('./_save_npy/keras48_1_validation_generator_y.npy')



# print(train_generator[0][0].shape, train_generator[0][1].shape)   # (9, 100, 100, 3) (9, 3)

# 2. 모델
# model.evaluate에 batch를 명시하지 않아왔지만 원래 batch_size가 존재했단 소리지.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
model = Sequential()
model.add(Conv2D( 64, (3,3), input_shape=(100, 100, 3)))
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
model.add(Dense(3, activation='softmax'))
# print(train_generator.describes)
# print(validation_generator.types)
# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # bc가 낮은거 metrics높은거 잡아주겠지??????
                                                                                       # optimizer='rmsprop'

# steps_per_epoch 함수로 넣는법
# aaa= len(dataset)
# batch=5
# spe=aaa/batch
# steps_per_epoch= spe

import os
path = "./_save/rps_npy_1.h5"
if os.path.exists(path):
    model.load_weights(path)
  #model = load_model(path)  
else:
    import time
    start = time.time()
    hist = model.fit(x_train, y_train, epochs=200, batch_size= 32, validation_split = 0.3)
    end = time.time()- start
    print("걸린시간 : ", round(end, 3), '초')
    model.save("./_save/rps_npy_1.h5")
    
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

import matplotlib.pyplot as plt
# print('loss : ', loss[-1]) # 마지막이 최종 것
# print('val_loss : ', val_loss[-1])
# print('accuracy : ', acc[-1])
# print('val_accuracy : ', val_acc[-1])

# epochs = range(1, len(loss)+1)

# # #plot keras13
# plt.figure(figsize=(9, 5))
# # plt.plot(hist.history['loss'])
# # plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# # plt.plot(hist.history['val_loss'], marker='.', c='blue', label='loss')
# # plt.plot(hist.history['acc'], marker='.', c='green', label='acc')
# # plt.plot(hist.history['val_acc'], marker='.', c='yellow', label='val_acc')

# plt.plot(epochs, loss, 'r--', label='loss')
# plt.plot(epochs, val_loss, 'r:', label="val_loss")
# plt.plot(epochs, acc, 'b--', label='acc')
# plt.plot(epochs, val_acc, 'b:', label='val_acc')

# plt.grid()
# # plt.title('loss')
# # plt.ylabel('loss')
# # plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()

# predict_generator takes your test data and gives you the output.
# evaluate_generator uses both your test input and output. It first predicts output using training input and then evaluates performance by comparing it against your test output. So it gives out a measure of performance, i.e. accuracy in your case.




# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/_predict/rps/'
sample_image = sample_directory + "scissors_yg.jpg"

# 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

# 샘플케이스 평가
loss, acc = model.evaluate(x_validation, y_validation)    # steps=5
#TypeError: 'float' object is not subscriptable
print("Between R.S.P Accuracy : ",str(np.round(acc ,2)*100)+ "%")# 여기서 accuracy는 이 밑의 샘플데이터에 대한 관측치가 아니고 모델 내에서 가위,바위,보를 학습하고 평가한 정확도임

image_ = keras_image.load_img(str(sample_image), target_size=(100, 100))
x = keras_image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /= 255.
# print(x)
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
print(classes)
y_predict = np.argmax(classes)#NDIMS
print(y_predict)


# print(classes)          # [[9.9968779e-01 3.1223366e-04 2.7075546e-09]]

#list combine
# xy_validation = list(zip(x_validation, y_validation))
# print(x_validation, y_validation)

#dict combine
# dict_x_validation = dict(enumerate(x_validation.flatten(), 1))
# dict_y_validation = dict(enumerate(x_validation.flatten(), 1))
# def merge_two_dicts(x, y):
#     z = x.copy()   # start with keys and values of x
#     z.update(y)    # modifies z with keys and values of y
#     return z
# dict_xy_validation = merge_two_dicts(dict_x_validation, dict_y_validation)



#numpy combine
# xy_validation.reset()
# xy_validation=np.concatenate((x_validation, y_validation), axis=0)

# print(dict_xy_validation)





# class_indices를 찍고 싶었는데.............
# {'paper': 0, 'rock': 1, 'scissors': 2}





if(y_predict==0):
    print(classes[0][0]*100, "의 확률로")
    print(" → '보'입니다. " )
elif(y_predict==1):
    print(classes[0][1]*100, "의 확률로")
    print(" → '바위'입니다. ")
elif(y_predict==2):
    print(classes[0][2]*100, "의 확률로")
    print(" → '가위'입니다. ")
else:
    print("ERROR")

# Between R.S.P Accuracy :  28.000000000000004%
# 99.96877908706665 의 확률로
#  → '보'입니다.
















###################################################### T R A S H ######################################################
# # # 4. 평가, 예측
# # # Evaluate
# # loss = model.evaluate_generator())

# # loss, acc = model.evaluate_generator(test_generator) #, steps=5????
# loss, acc = model.evaluate(validation_generator) #, steps=5????
# # print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
# print("Accuracy : ", str(np.round(acc[:2],2)*100)+ "%")

# # # Predict
# # 모델 사용 시에 제네레이터에서 제공되는 샘플을 입력할 때는 predict_generator 함수를 사용합니다. 예측 결과는 클래스별 확률 벡터로 출력되며, 클래스에 해당하는 열을 알기 위해서는 제네레이터의 ‘class_indices’를 출력하면 해당 열의 클래스명을 알려줍니다.
# validation_generator.reset()
# # output = model.predict_generator(test_generator)
# output = model.predict(validation_generator[:2])
# y_predict = np.argmax(output)#NDIMS

# # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# print(validation_generator.class_indices)
# print("예측값 : ", y_predict)


# 2021-12-27 20:13:45.592475: F .\tensorflow/core/framework/tensor.h:806] Check failed: NDIMS == new_sizes.size() (2 vs. 1)
# 오류
# https://stackoverflow.com/questions/50319749/tensorflow-check-failed-ndims-new-sizes-size-2-vs-1

