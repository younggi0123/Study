from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
import numpy as np



# 실습 !

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000, 10)
# print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000, 10)
y_train = to_categorical(y_train)       # y값은 카테고리컬 해줘여쥐
y_test = to_categorical(y_test)         # test도 카테고리컬해줘야아아아앆!!!!!!!!!!!!!!!!!!!!!!!

# 2. 모델구성
model  =  Sequential()
model.add(Conv1D(64, 2, input_shape=(28,28) ))
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





import time
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
          validation_split=0.3, callbacks=[es])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')



# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( x_test )
#print("예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict) #ypredict test비교
print('r2스코어 : ', r2)


# Conv1D 수행 시
# 걸린시간 :  40.201 초
# accuracy: 0.8403
# loss :  [0.4644117057323456, 0.8403000235557556]
# r2스코어 :  0.7448364390362626