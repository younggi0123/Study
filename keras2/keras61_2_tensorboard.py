# 【       Subject : 22'. 02. 07. Tensorboard       】


# 시각화에 강한 강점을 가지고 있음
# 스칼라 형식 , 그래프, 만든 모델을 UML화 해주는 기능 까지 있음

# 127.0.0.1 이란 본인의 고유한 LOCAL HOST임
# localhost:6006대신 127.0.0.1:6006로 사용해도 됨

# web상에서 이 기능에대해 딱 명시해주는것텐서플로에서 제공함
# 시각화 등에 이쁘게 볼 수 있게 해준다


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten # 1D는 선만 그어. 2D부터 이미지
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

scaler = StandardScaler()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)         # test도 카테고리컬해줘야 10으로 바뀜
print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000, )
print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000, )
#reshape또한 일종의 전처리이다.


n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1

x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)


# x_train = x_train.reshape(60000, 28, 28, 1) # 전체를 다 곱했을때 같으면 상관없다
# x_test = x_test.reshape(60000, 28, 14, 1) # 이것도 가능하다
# x_test = x_test.reshape(10000, 28, 28, 1)
                                            # 위치는 바뀌지 않는다
                                            # convolutionlayer로 넣기 위해 4차원으로 바꾼것
print(x_train.shape)                        # (60000,28,28,1)

print(np.unique(y_train, return_counts=True))   #10개 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#input data shape rate : 60000,20 4차원

# print(y)
# print(y.shape)
print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,)

# 2. 모델구성
#hint conv-layer는 3~4개
model  =  Sequential() 
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(28, 28, 1 ) ))
model.add(Conv2D(5, (2,2), activation="relu") )
model.add(Dropout(0.2))
model.add(Conv2D(7, (2,2), activation="relu") )
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation='softmax'))




# 3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(lr=learning_rate)


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    


from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)




#################################################################추가된 내용##########################################################################

from tensorflow.keras.callbacks import TensorBoard
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0, write_graph=True, write_images=True)     # 경로잡기 중요! #histogram_freq= 넣어줘야.
                                                                                                      # 파라미터는 알아서 찾아보기 !

######################################################################################################################################################



import  time
start = time.time()
hist = model.fit(x_train, y_train, epochs=16, batch_size=32, verbose=1,
          validation_split=0.3, callbacks=[es, reduce_lr, tb]   )
end = time.time()
# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)





print("#########################################################################################################")
print('lerning rate : ', learning_rate)
print('loss : ', round(loss[0],4))
print('accuracy : ', round(loss[1], 4))
print("걸린시간 : ", round(end))


# 기존 loss :  0.0760224387049675 accuracy :  0.9810000061988831

#########################################################################################################
# 적용후
# lerning rate :  0.01
# loss :  0.0939
# accuracy :  0.9776
# 걸린시간 :  1644214447


################################################## 시각화 ##################################################
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))



#1.
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title(' LOSS ')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2.
plt.subplot(2,1,2)
plt.plot(hist.history['accuracy'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue', label='val_accuracy')
plt.grid()
plt.title(' ACC ')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])



plt.show()











'''

로컬호스트 웹서버 구동하는 법 (DOS상에서 작업!)

(현재경로에 train생성되고)
마치 웹 서버처럼 구동해주고 내 컴퓨터의 6006으로 접속하란말
텐서보드에 바로 접속이 가능하다/.


in CMD

C:\WINDOWS\system32>d:

D:\>cd study

D:\Study>cd _save

D:\Study\_save>cd _graph

D:\Study\_save\_graph>_graph 디렉터리
'_graph'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는
배치 파일이 아닙니다.

D:\Study\_save\_graph>_graph>dir/w
지정된 경로를 찾을 수 없습니다.

D:\Study\_save\_graph>dir/w
 D 드라이브의 볼륨: 새 볼륨
 볼륨 일련 번호: 5C42-445B

 D:\Study\_save\_graph 디렉터리
[.]  [..]
               0개 파일                   0 바이트
               2개 디렉터리  793,370,570,752 바이트 남음

D:\Study\_save\_graph>tensorboard --logdir=.


웹창에 로컬호스트 치면 접속됨

'''