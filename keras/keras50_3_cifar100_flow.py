# keras50 동일_
# 훈련데이터 10만개로 증폭
# 완료후 기존 모델과 비교
# save_dir도  _tmp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제
# 예를들어 여자 900 남자 100의 데이터를 ACCURACY 로 할 경우 망가지니까 F1-SCORE로 ㄱㄱ
# 이런식으로 통계적 지식 가미


from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

train_datagen = ImageDataGenerator(  
    rescale=1./255, 
    horizontal_flip = True,
    #vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    #rotation_range= 5,
    zoom_range = 0.1,
    #shear_range = 0.7,
    fill_mode= 'nearest')

test_datagen = ImageDataGenerator(  
    rescale=1./255
    )


print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)#3이 아닌건 아직 카테고리컬을 안해서
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# 증폭 데이터 생성
augment_size = 50000
randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()  
y_augmented = y_train[randidx].copy() 

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3)
x_train = x_train.reshape(x_train.shape[0],32,32,3)
x_test = x_test.reshape(x_test.shape[0],32,32,3)





# 자리변경
# x_augmented = train_datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle= False).next()[0]
x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle= False, save_to_dir='../_temp/').next()[0]

                                                    # x는 [0]에 있고 y는 [1]에 있어서 마지막에 [0]을 붙임으로서 x만 뽑아줌
                                                    # [0]으로 하면 shape(12000, 28, 28, 1) [1]로 하면 (12000,)

# xy_test = test_datagen.flow(x_test,y_test,batch_size=32)

x_train = np.concatenate((x_train, x_augmented))  # 결과 : (100000, 32, 32, 3)
y_train = np.concatenate((y_train, y_augmented))  # 결과 : (100000,)
print(x_train.shape, y_train.shape) # (100000, 32, 32, 3) (100000, 1)


# print(x_test.ndim)
# print(y_test.ndim)


# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)       # y값은 카테고리컬 해줘여쥐
# y_test = to_categorical(y_test)         # test도 카테고리컬해줘야아아아앆!!!!!!!!!!!!!!!!!!!!!!!




# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(32,32,3) ))
model.add(Conv2D(64, (2,2) ))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))#10인거 보려면 카테고리컬 찍어보면.

# generator flow를 통과한 tuple상태를 fit_generator로
# 3. 컴파일, 훈련

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # one-hot-encoding이 귀찮아서 sparse로.

# print(len(xy_train[0]))     # 1250

# 가중치 저장
import os
path = "./_save/keras50_3.h5"
print(os.getcwd())
if os.path.exists(path):
    model.load_weights(path)
else:
    import time
    start = time.time()
    
    hist = model.fit(x_train,y_train, epochs=10, batch_size=64,#batch will same with first layer input shape
                    validation_split=0.2)
    end = time.time()- start
    print("걸린시간 : ", round(end, 3), '초')
    model.save("./_save/keras50_3.h5")




# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
predict = model.predict( x_test )

predict = np.argmax(predict, axis=1)
# y_test = np.argmax(y_test, axis=1)

# ※☆※★※☆※★※☆※★ fasion mnist에서 r2스코어가 왜 나와아아아앆!!!!!!!!!!!!!!!!!!!!!??? accuracy score로 빼!!!!!!!!!!!!!!!!!!!!!!!!!!
# 참고 :https://leedakyeong.tistory.com/entry/%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EC%84%B1%EB%8A%A5-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C-Confusion-Matrix%EB%9E%80-%EC%A0%95%ED%99%95%EB%8F%84Accuracy-%EC%A0%95%EB%B0%80%EB%8F%84Precision-%EC%9E%AC%ED%98%84%EB%8F%84Recall-F1-Score
from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test,predict) #ypredict test비교
print('accuracy_score : ', accuracy_score)

# loss :  [nan, 0.012000000104308128]
# accuracy_score :  0.012

