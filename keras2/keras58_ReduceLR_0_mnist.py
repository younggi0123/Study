# keras/mnist 30-2 기반 optimizer 변형( optimizer& learning rate 변형)

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


#################################################################추가된 내용##########################################################################
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.01)
######################################################################################################################################################



model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

import  time
start = time.time()
model.fit(x_train, y_train, epochs=16, batch_size=32, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
          validation_split=0.3, callbacks=[es])
end = time.time()
# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', round(loss[0],4))             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', round(loss[1],4))              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★
print("걸린시간 : ", round(end - start,4))




#           [Learning Rate 성능비교]

#       Case of Learning rate 0.01 ▼
            # loss :  0.1547
            # accuracy :  0.9649
            # 걸린시간 :  143.1483

#       Case of Learning rate 0.001         ⓥ Check !
            # loss :  0.0711
            # accuracy :  0.9844
            # 걸린시간 :  144.4337

#       Case of Learning rate 0.0001
            # loss :  0.0829
            # accuracy :  0.9752
            # 걸린시간 :  148.5832



