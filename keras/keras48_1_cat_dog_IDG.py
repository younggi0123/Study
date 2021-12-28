# kaggle.com/c/dogs-vs-cats/data
# 데이터수 8005개
# 이부분 블로그 글 별로 없으니 ㄱㄱㄱㄱ

# batch 딱 안떨어진다? 소수점에 +1하면 됨

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image


# IDG를 정의한다.
train_datagen = ImageDataGenerator(
    rescale = 1./255,               # 데이터 픽셀 범위 0과 1사이로 scaling하기 위한 인자
    horizontal_flip = False,         # 상하반전(mnist 데이터 등에서 숫자예측시 6과 9는 다른 숫자가 되므로 유의)
    vertical_flip =False,            # 좌우반전
    width_shift_range = 0.1,        # 좌우이동
    height_shift_range = 0.1,       # 상하이동
    # rotation_range = 5,             # 회전이동
    # zoom_range = 1.2,               # zoom 증폭
    # shear_range = 0.7,              # 부동소수점. 층밀리기의 강도입니다. (도 단위의 반시계 방향 층밀리기 각도)
    fill_mode = 'nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# D:\_data\image\cat_dog\training_set\cats
training_set = train_datagen.flow_from_directory(
    '../_data/image/cat_dog/training_set/',
    target_size = (100, 100),
    batch_size = 55,
    class_mode = 'binary',
    shuffle = True
)

test_set = test_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set/',
    target_size = (100, 100),
    batch_size = 55,
    class_mode = 'binary',
    shuffle = False,
)

# dogs_train = train_datagen.flow_from_directory(
#     '../_data/image/cat_dog/training_set/',
#     target_size = (100, 100),
#     batch_size = 5,
#     class_mode = 'binary',
#     shuffle = True
# )

# dogs_test = test_datagen.flow_from_directory(
#     '../_data/image/cat_dog/test_set/',
#     target_size = (100, 100),
#     batch_size = 5,
#     class_mode = 'binary',
#     shuffle = False,
# )

# print(training_set[0][0].shape, training_set[0][1].shape)  (55, 100, 100, 3) (55,)

# 데이터 구조 파악
# print(type(cats_train))
# print(type(cats_train[0]))
# print(type(cats_train[0][0]))
# print(type(cats_train[0][1]))



# 2. 모델
# model.evaluate에 batch를 명시하지 않아왔지만 원래 batch_size가 존재했단 소리지.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
model = Sequential()
model.add(Conv2D( 55, (2,2), input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=2, strides=1, padding="VALID"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))




# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # bc가 낮은거 metrics높은거 잡아주겠지??????

import os
path = "./_save/cat_dog_IDG_1.h5"
if os.path.exists(path):
    model.load_weights(path)
  #model = load_model(path)  
else:
    import time
    start = time.time()
    hist = model.fit_generator(training_set, epochs=10, steps_per_epoch=training_set.samples//91,
                           validation_data= test_set,
                           validation_steps= test_set.samples//91,
                           )
    end = time.time()- start
    print("걸린시간 : ", round(end, 3), '초')
    model.save("./_save/cat_dog_IDG_1.h5")
    
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

import matplotlib.pyplot as plt


# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/_predict/cat_dog/'
sample_image = sample_directory + "younggi.jpg"

# 샘플 케이스 확인
image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

# 샘플케이스 평가
loss, acc = model.evaluate(test_set)    # steps=5
#TypeError: 'float' object is not subscriptable
print("Between cat and dog Accuracy : ",str(np.round(acc ,2)*100)+ "%")# 여기서 accuracy는 이 밑의 샘플데이터에 대한 관측치가 아니고 모델 내에서 가위,바위,보를 학습하고 평가한 정확도임

image_ = keras_image.load_img(str(sample_image), target_size=(100, 100))
x = keras_image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /= 255.
# print(x)
images = np.vstack([x])

# classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS
# print(classes)

y_predict=model.predict(images, batch_size=40)

# print(type(validation_generator))#DirectoryIterator

test_set.reset()
print(test_set.class_indices)
# class_indices
#  {'cats': 0, 'dogs': 1}


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

# {'cats': 0, 'dogs': 1}
# Between cat and dog Accuracy :  50.0%

# 49.85 %의 확률로
#  → '고양이'입니다.


# 답이 없는 문제(인간이니까)

































# hist = model.fit_generator(training_set, epochs=20, steps_per_epoch=91,    # batch_size를 명시할 필요가 없는 대신에 epo당 step을 몇 번 할지를 명시 한다.
#                                                                        # 전체데이터 나누기 batchsize( Total/Batch = 8005/55= 91 ) => ※무조건 써줘야 함※
#                                                                        # 전체데이터는 경로 들가서 몇 개인지 ㄱㄱ 여기선 80,80개였음
#                            validation_data=test_set,
#                            validation_steps= 4,
#                            # verbose = 1
#                            )



# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# import matplotlib.pyplot as plt
# print('loss : ', loss[-1]) # 마지막이 최종 것
# print('val_loss : ', val_loss[-1])
# print('accuracy : ', acc[-1])
# print('val_ accuracy : ', val_acc[-1])

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
# # plt.title('측정치 그래프')
# # plt.ylabel('loss,acc,val_loss,val_acc')
# # plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()



# # 개 고양이 정확도 측정
# # loss :  0.6931890845298767
# # val_loss :  0.6921833157539368
# # accuracy :  0.4855421781539917
# # val_ accuracy :  1.0











# # # 4. 평가, 예측
# # # Evaluate
# # loss = model.evaluate_generator(test_set)
# # print('loss : ', loss)
# # # Predict
# # y_predict = model.predict_generator(test_set)
# # #print("예측값 : ", y_predict)
