# 【       Subject : 21'. 12. 01. keras19_1~7까지의 7종류 파일 데이터에 4개지 전처리 방법을 적용해본다.       】

# File 4. Iris Data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
import numpy as np

# 1. 데이터
# Data load
datasets = load_iris()
x = datasets.data
y = datasets.target
y = to_categorical(y)
# print(y.shape)      #(150, 3)
# print(x.shape, y.shape)                  # (150, 4)(150, )

# 최소값 최대값
# print(np.min(x), np.max(x))     #  0.1 7.9

x= x.reshape(150,4,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


# print(np.unique(y))             # 값을 찍어서 어떤 분류확인 : [0 1 2]이다. 다중분류니까 Softmax & categorical_Crossentropyㄱㄱ

# 2. 모델링
# Model Set
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(4,1)))         #input dim 4될 것
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='linear'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))               # ★ activation은 softmax ★ activation은 softmax ★ activation은 softmax ★

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=True)    



import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
          validation_split=0.2, callbacks=[es])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')


# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# results = model.predict(x_test[:7])
# print(y_test[:7])
# print(results)

# 55.588287353515625 r2 스코어 :  0.3271576401020574

# Conv1D 수행 시
# 걸린시간 :  44.44 초
# loss :  0.06216948851943016
# accuracy :  0.9666666388511658