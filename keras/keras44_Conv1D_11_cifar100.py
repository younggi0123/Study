from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
import numpy as np
import time


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

y_train = to_categorical(y_train)       # y값은 카테고리컬 해줘여쥐
y_test = to_categorical(y_test)         # test도 카테고리컬해줘야아아아앆!!!!!!!!!!!!!!!!!!!!!!!
print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 100)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 100)

# reshape
x_train = x_train.reshape(50000,96,32)
x_test = x_test.reshape(10000,96,32)
print(x_train.shape, y_train.shape)     # (50000, 96, 32) (50000, 100)
print(x_test.shape, y_test.shape)       # (10000, 96, 32) (10000, 100)


# 2. 모델구성
model  =  Sequential()
model.add(Conv1D(64, 2, input_shape=(96,32)) )
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation='softmax'))



# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    



import time
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
          validation_split=0.3, callbacks=[es])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')




# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★



# loss :  4.6056084632873535    accuracy :  0.009999999776482582



# Conv1D 수행 시
# 걸린시간 :  34.017 초
# loss :  4.803642272949219
# accuracy :  0.015699999406933784