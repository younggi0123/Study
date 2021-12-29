# 내건 안돌아감.......


# 넘파이 쉐이프 확실히 하자....................................
# 고통의 무한 오류 AxisError: axis 1 is out of bounds for array of dimension 1
#https://everyday-image-processing.tistory.com/87



# ☆★ 참 고 ★☆
###################'sparse_categorical_crossentropy'###################
# ☆★https://circle-square.tistory.com/108★☆
# 49-2 copy
# 모델링 구성 부분

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image as keras_image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation

import warnings
warnings.filterwarnings('ignore')

import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print("xtest는 :", x_test.shape)#(10000, 28, 28)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    fill_mode = 'nearest'
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])                 # 60000
print(randidx)                          # [19388 40444  5836 ... 51885 13813  7103]
print(np.min(randidx), np.max(randidx)) # 1 59998

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape)                # (40000, 28, 28)   # 40000개 들어가고, shape는 28, 28이 들어가겠다.

# print(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2]) # 각 40000    28     28
print(x_test.shape[0])#10000

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)
x_train = x_train.reshape(60000, 28, 28, 1)
# x_train = x_train.reshape(40000, 42, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


xy_train = train_datagen.flow(x_augmented, y_augmented,
                              batch_size=32, # augment_size,
                              shuffle=False
                              ) # .next()[0]


###추가
xy_test = train_datagen.flow(x_test, y_test,
                              batch_size=32, # augment_size,
                              shuffle=False
                              ) # .next()[0]
print("하..")
# print(xy_train.shape)
print("후./...")
# print(xy_test.shape)


# print(y_train.shape)
# print(y_augmented.shape)

# 꼭 봐
# https://everyday-image-processing.tistory.com/86
# x_train = np.concatenate((x_train, np.transpose(x_augmented)),axis=1)#[[[[0].....
# y_train = np.concatenate((y_train, np.transpose(y_augmented)),axis=1)#[0 2 1 ... 7 1 7]


# x_train = np.concatenate((x_train, x_augmented),axis=1)#[[[[0].....
# y_train = np.concatenate((y_train, y_augmented),axis=1)#[0 2 1 ... 7 1 7]


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1) ))
model.add(Conv2D(64, (2,2) ))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

# generator flow를 통과한 tuple상태를 fit_generator로
# 3. 컴파일, 훈련

###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################
###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################
###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################
###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################
###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################
###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################
###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################'sparse_categorical_crossentropy'###################

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # one-hot-encoding이 귀찮아서 sparse로.

# print(len(xy_train[0]))     # 1250

# 가중치 저장
import os
path = "./_save/keras49_6_flow.h5"
print(os.getcwd())
if os.path.exists(path):
    model.load_weights(path)
else:
    import time
    start = time.time()
    
    #fit
    model.fit_generator( xy_train, epochs=10, steps_per_epoch=len(xy_train) )   #//augment_size ) 나누는거???
    
    end = time.time()- start
    print("걸린시간 : ", round(end, 3), '초')
    model.save("./_save/keras49_6_flow.h5")
    



# 4. 평가, 예측
# Evaluate
loss = model.evaluate(xy_test)
print('loss : ', loss)
# Predict
xy_predict = model.predict( xy_test )

xy_predict = np.argmax(xy_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

# ※☆※★※☆※★※☆※★ fasion mnist에서 r2스코어가 왜 나와아아아앆!!!!!!!!!!!!!!!!!!!!!??? accuracy score로 빼!!!!!!!!!!!!!!!!!!!!!!!!!!
# 참고 :https://leedakyeong.tistory.com/entry/%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EC%84%B1%EB%8A%A5-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C-Confusion-Matrix%EB%9E%80-%EC%A0%95%ED%99%95%EB%8F%84Accuracy-%EC%A0%95%EB%B0%80%EB%8F%84Precision-%EC%9E%AC%ED%98%84%EB%8F%84Recall-F1-Score
from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, xy_predict) #ypredict test비교
print('accuracy_score : ', accuracy_score)

























############################################### T R A S H ###############################################
# print(xy_train)# x&y가 합쳐진 모습
# print(xy_train[0].shape, xy_train[1].shape)
# print(x_train)
# print("뀨")
# print(y_train)#[9 0 0 ... 3 0 5]
# print("뀨우우우")
# print(xy_train[0])#, array([3, 0, 2, 1, 5, 4, 0, 3, 3, 2, 8, 0, 7, 9, 7, 8, 5, 3, 9, 1, 6, 3, 5, 7, 5, 7, 1, 9, 1, 8, 2, 1], dtype=uint8))
# print("뀨우우우뀨우우우")
# print(xy_train[1])#, array([3, 0, 0, 4, 5, 9, 7, 1, 7, 7, 0, 3, 8, 8, 8, 4, 9, 2, 7, 8, 1, 4, 9, 8, 3, 3, 9, 1, 8, 2, 8, 6], dtype=uint8))
# print("뀨우우우뀨우우우뀨우우우")
# print(x_augmented)
# print(y_augmented)#[2 3 3 ... 2 4 8]