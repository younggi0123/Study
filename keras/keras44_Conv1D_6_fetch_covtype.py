from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape)      # (581012, 54)
print(y.shape)      # (581012, )

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y.reshape(-1,1))

x= x.reshape(581012, 54, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#2. 모델 구성

model = Sequential()
model.add(Conv1D(10,2, input_shape=(54, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k35_6_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=5000, validation_split=0.2, callbacks=[es, mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')



#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

# loss :  36.39024353027344 r2 스코어 :  0.5595313606671943


# Conv1D 수행 시
# 걸린시간 :  12.023 초
# loss :  0.7057424783706665
# accurcy :  0.6839544773101807