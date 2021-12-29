# ☆★ 참 고 ★☆
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


import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


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

x_augmented = x_train[randidx].copy()   # 
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)                # (40000, 28, 28)   # 40000개 들어가고, shape는 28, 28이 들어가겠다.
print(x_augmented.shape)                # (40000, )

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size = augment_size, shuffle=False,
                                 ).next()[0]

print(x_augmented)
print(x_augmented.shape)                    # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))      #(100000, 28, 28, 1)
y_train = np.concatenate((y_train, y_augmented))

print(x_train)
print(x_train.shape)








############################################## keras49_3_flow에서 모델 구성부분 추가 ##############################################
# 카테고리컬...?!
y_train = to_categorical(y_train)       # y값은 카테고리컬 해줘여쥐
y_test = to_categorical(y_test)         # test도 카테고리컬해줘야아아아앆!!!!!!!!!!!!!!!!!!!!!!!

# 2. 모델구성
model  =  Sequential()
model.add(Conv2D(64, 2, input_shape=(28,28,1) ))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))



# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    



import os
path = "./_save/keras49_3_flow_model.h5"
print(os.getcwd())
if os.path.exists(path):
    model.load_weights(path)
else:
    import time
    start = time.time()

    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
          validation_split=0.3, callbacks=[es])

    end = time.time()- start
    print("걸린시간 : ", round(end, 3), '초')
    model.save("./_save/keras49_3_flow_model.h5")
    



# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( x_test )


#ValueError: Classification metrics can't handle a mix of multilabel-indicator and continuous-multioutput targets
#https://www.google.com/search?q=+ValueError%3A+Classification+metrics+can%27t+handle+a+mix+of+multilabel-indicator+and+continuous-multioutput+targets&newwindow=1&sxsrf=AOaemvLCSzQj_zePWc3pSgENd4spIWlh-A%3A1640745928919&ei=yMvLYf7KN_aHxc8PodCmiAg&ved=0ahUKEwj-zony_of1AhX2Q_EDHSGoCYEQ4dUDCA4&uact=5&oq=+ValueError%3A+Classification+metrics+can%27t+handle+a+mix+of+multilabel-indicator+and+continuous-multioutput+targets&gs_lcp=Cgdnd3Mtd2l6EANKBAhBGABKBAhGGABQAFgAYM4LaABwAHgAgAEAiAEAkgEAmAEAoAECoAEBwAEB&sclient=gws-wiz
#https://stackoverflow.com/questions/48987959/classification-metrics-cant-handle-a-mix-of-continuous-multioutput-and-multi-la/48991515

#Answer)
# To sum this up: with this code you should get your matrix
# y_pred=model.predict(X_test) 
# y_pred=np.argmax(y_pred, axis=1)
# y_test=np.argmax(y_test, axis=1)
# cm = confusion_matrix(y_test, y_pred)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
#print("예측값 : ", y_predict)




# ※☆※★※☆※★※☆※★ fasion mnist에서 r2스코어가 왜 나와아아아앆!!!!!!!!!!!!!!!!!!!!!??? accuracy score로 빼!!!!!!!!!!!!!!!!!!!!!!!!!!
# 참고 :https://leedakyeong.tistory.com/entry/%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EC%84%B1%EB%8A%A5-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C-Confusion-Matrix%EB%9E%80-%EC%A0%95%ED%99%95%EB%8F%84Accuracy-%EC%A0%95%EB%B0%80%EB%8F%84Precision-%EC%9E%AC%ED%98%84%EB%8F%84Recall-F1-Score

from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_predict) #ypredict test비교
print('accuracy_score : ', accuracy_score)


# 기존 keras44_fashion_mnist 에서 Conv1D 수행 시
# 걸린시간 :  40.201 초
# accuracy: 0.8403 ⓥ
# loss :  [0.4644117057323456, 0.8403000235557556]
# r2스코어 :  0.7448364390362626





# new버전
# loss :  [0.46839258074760437, 0.8629000186920166]
# accuracy_score :  0.8629 ⓥ