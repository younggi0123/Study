import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import time  # 일정 라인이 가동되는 시간을 측정하는 함수

#1. 데이터 정제
datasets = load_boston()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)


#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dropout(0.2))     #dropout은 노드마다 적용가능하다
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(3))
model.add(Dropout(0.1))
model.add(Dense(1))


#3. 컴파일
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")    # ex) 1206_0456(12월 06일 4시 56분)
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, 'k26_', datetime, '_', filename])
'''
model_path 예시 : ./ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf5
'''

es = EarlyStopping(mode='min', patience=50, monitor='val_loss', verbose=1) #, restore_best_weights=True
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path)

start = time.time()
hist =model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
          validation_split=0.3, callbacks=[es, mcp])
end = time.time() - start

print("걸린 시간 : ", round(end, 2), '초')


model.save('./_save/keras26_4_save_model.h5') 
# # model = load_model('./_save/keras26_1_save_model.h5')
# # model = load_model('./_ModelCheckPoint/keras26_1_MCP.hdf5')



print("======================= 1. 기본출력 =======================")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict(x_test)
r2 = r2_score(y_test, result)
print("r2 스코어 : ", r2)

print("====================== 2. load_model 출력 =======================")
model2 = load_model('./_save/keras26_4_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print("loss : ", loss2)

result2 = model2.predict(x_test)
r2 = r2_score(y_test, result2)
print("r2 스코어 : ", r2)

print("====================== 3. ModelCheckPoint 출력 =======================")
model3 = load_model('_ModelCheckPoint/keras26_1_MCP.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print("loss : ", loss3)

result3 = model3.predict(x_test)
r2 = r2_score(y_test, result3)
print("r2 스코어 : ", r2)

#드랍아웃 적용시 미적용시
