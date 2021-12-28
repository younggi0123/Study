# kaggle.com/c/dogs-vs-cats/data
# 데이터수 2520개
# ★https://necromuralist.github.io/Neurotic-Networking/posts/keras/rock-paper-scissors/★

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# IDG를 정의한다.
train_datagen = ImageDataGenerator(
    rescale = 1./255,               # 데이터 픽셀 범위 0과 1사이로 scaling하기 위한 인자
    horizontal_flip = False,         # 상하반전(mnist 데이터 등에서 숫자예측시 6과 9는 다른 숫자가 되므로 유의)
    vertical_flip =False,            # 좌우반전
    # width_shift_range = 0.1,        # 좌우이동
    # height_shift_range = 0.1,       # 상하이동
    # rotation_range = 5,             # 회전이동
    # zoom_range = 1.2,               # zoom 증폭
    # shear_range = 0.7,              # 부동소수점. 층밀리기의 강도입니다. (도 단위의 반시계 방향 층밀리기 각도)
    # validation_split=0.2,       # ★
    fill_mode = 'nearest'
)

# ☆★☆★☆★☆★ https://naenjun.tistory.com/17 ☆★☆★☆★☆★
#train-test-valdiation set
train_generator = train_datagen.flow_from_directory(
    '../_data/image/rps/',
    target_size = (100, 100),
    batch_size = 9,
    class_mode = 'categorical', # categorical쓰면 one-hot 안 해도됨
    subset='training',
    shuffle = True
)

# D:\_data\image\_predict
test_datagen = ImageDataGenerator(
    rescale=1./255
)
# D:\_data\image\_predict
test_generator = test_datagen.flow_from_directory(
    '../_data/image/_predict/',
    target_size = (100, 100),
    batch_size = 9,
    class_mode = 'categorical',
    shuffle = False,
)



# print(train_generator[0][0].shape, train_generator[0][1].shape)   # (9, 100, 100, 3) (9, 3)

# 2. 모델
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
model.add(Dense(3, activation='softmax'))
print(train_generator.describes)
print(test_generator.types)
# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # bc가 낮은거 metrics높은거 잡아주겠지??????
hist = model.fit_generator(train_generator, epochs=5, steps_per_epoch = train_generator.samples//9 ,
                           validation_data= test_generator,
                           validation_steps= test_generator.samples//9,
                           )


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

# # 4. 평가, 예측
# # Evaluate
# loss = model.evaluate_generator())

# loss, acc = model.evaluate_generator(test_generator) #, steps=5????
loss, acc = model.evaluate(test_generator) #, steps=5????
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
print("Accuracy : ", str(np.round(acc[:2],2)*100)+ "%")

# # Predict
# 모델 사용 시에 제네레이터에서 제공되는 샘플을 입력할 때는 predict_generator 함수를 사용합니다. 예측 결과는 클래스별 확률 벡터로 출력되며, 클래스에 해당하는 열을 알기 위해서는 제네레이터의 ‘class_indices’를 출력하면 해당 열의 클래스명을 알려줍니다.
test_generator.reset()
# output = model.predict_generator(test_generator)
output = model.predict(test_generator[:2])
y_predict = np.argmax(output)#NDIMS

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print("예측값 : ", y_predict)


# 2021-12-27 20:13:45.592475: F .\tensorflow/core/framework/tensor.h:806] Check failed: NDIMS == new_sizes.size() (2 vs. 1)
# 오류
# https://stackoverflow.com/questions/50319749/tensorflow-check-failed-ndims-new-sizes-size-2-vs-1