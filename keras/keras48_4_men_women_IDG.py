# 과제
# 본인 사진으로 predict 하시오
# d:/_data안에 넣고.


# kaggle.com/c/dogs-vs-cats/data
# 데이터수 1027개


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
    validation_split=0.2,
    fill_mode = 'nearest'
)
# D:\_data\image\men_women\data

# test_datagen = ImageDataGenerator(
#     rescale=1./255
# )
# ☆★☆★☆★☆★ https://naenjun.tistory.com/17 ☆★☆★☆★☆★
#train-test-valdiation set
train_generator = train_datagen.flow_from_directory(
    '../_data/image/men_women/data',
    target_size = (100, 100),
    batch_size = 79,
    class_mode = 'binary',
    subset='training',
    shuffle = True
)#Found 2648 images belonging to 2 classes.

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/men_women/data',
    target_size = (100, 100),
    batch_size = 79,
    class_mode = 'binary',
    subset='validation',
    shuffle = True
)#Found 661 images belonging to 2 classes.

print(train_generator[0][0].shape, train_generator[0][1].shape)   # (79, 100, 100, 3) (79,)

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
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # bc가 낮은거 metrics높은거 잡아주겠지??????
                                                                                       # optimizer='rmsprop'

import os
path = "./_save/men_women_IDG_1.h5"
if os.path.exists(path):
    model.load_weights(path)
  #model = load_model(path)  
else:
    import time
    start = time.time()
    hist = model.fit_generator(train_generator, epochs=10, steps_per_epoch=train_generator.samples//79,
                           validation_data= validation_generator,
                           validation_steps= validation_generator.samples//79,
                           )
    end = time.time()- start
    print("걸린시간 : ", round(end, 3), '초')
    model.save("./_save/men_women_IDG_1.h5")
    
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

import matplotlib.pyplot as plt


# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/_predict/men_women/'
sample_image = sample_directory + "younggi.jpg"

# 샘플 케이스 확인
image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

# 샘플케이스 평가
loss, acc = model.evaluate(validation_generator)    # steps=5
#TypeError: 'float' object is not subscriptable
print("Between men and women Accuracy : ",str(np.round(acc ,2)*100)+ "%")# 여기서 accuracy는 이 밑의 샘플데이터에 대한 관측치가 아니고 모델 내에서 가위,바위,보를 학습하고 평가한 정확도임

image_ = keras_image.load_img(str(sample_image), target_size=(100, 100))
x = keras_image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /= 255.
# print(x)
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
y_predict = np.argmax(classes)#NDIMS
# print(classes)          # [[0.4892104]]

# print(type(validation_generator))#DirectoryIterator

validation_generator.reset()
print(validation_generator.class_indices)
# class_indices
#  {'men': 0, 'women': 1}


print("\n")
if(y_predict>=0.5):
    person= classes[0][0]*100
    print(round( person, 2), "%의 확률로")
    print(" → '여성'입니다. " )
elif(y_predict<0.5):
    horse= (100-(classes[0][0]*100))
    print(round( horse, 2), "%의 확률로")
    print(" → '남성'입니다. ")
else:
    print("ERROR 발생")
    
# Between men and women Accuracy :  48.0%

# 51.08 %의 확률로
#  → '남성'입니다.    
# 정답.
    
    
    
#오류찾기##오류찾기##오류찾기##오류찾기##오류찾기##오류찾기##오류찾기##오류찾기##오류찾기#
# print("\n")
# if(y_predict>=0.5 and y_predict<1):
#     person= classes[0][0]*100
#     print(round( person, 2), "%의 확률로")
#     print(" → '여성'입니다. " )
# elif(y_predict<0.5 and y_predict>0):
#     horse= (100-(classes[0][0]*100))
#     print(round( horse, 2), "%의 확률로")
#     print(" → '남성'입니다. ")
# else:
#     print("ERROR 발생")
