from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from pandas import Series, DataFrame

#1. 데이터 
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape) # 178 13

x =x.reshape(178,13,1)


# print(type(xx))
# print(xx.corr())
# xx['quality'] = y
# import matplotlib.pyplot as plt
# import seaborn as sns 
# plt.figure(figsize=(10,10))
# sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)
# print(x_train.shape) # (124,12)
# print(x_test.shape)  # (54,12)


#2. 모델구성
model = Sequential() 
model.add(LSTM(7, input_shape = (13,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
# print(y_train.shape) 
# print(y_test.shape)



#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) 
print('loss:', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
print(y_test.shape, y_predict.shape)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# 55.588287353515625 r2 스코어 :  0.3271576401020574
