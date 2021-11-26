#  13_1이 어떻다datetime A combination of a date and a time. Attributes: ()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import validation
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import time
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
# Data load
datasets = load_diabetes()
# Train_set
x = datasets.data
y = datasets.target
#print(x.shape) #feature=13
#print(y.shape) #feature= 1

# Train&test&val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=66)

# 2. 모델구성
# Model Set
model = Sequential()
# Model Add
# model.add(Dense(70, input_dim=13))
# model.add(Dense(55))
# model.add(Dense(40))
# model.add(Dense(25))
# model.add(Dense(10))
# model.add(Dense(1))
model.add(Dense(100, input_dim=10))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
start = time.time()
# History <- Fit
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2)
end = time.time() - start

print("걸린시간 : ", round(end))

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( x_test )
#print("예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict) #ypredict test비교
print('r2스코어 : ', r2)

# print("===============================================")
# print(hist)
# print("===============================================")
# print(hist.history)        # key-value값의 dictionary형태
#                            # model.fit에서 loss와val_loss값을 반환해준다
# print("===============================================")
# print(hist.history['loss']) 
# print("===============================================")
# print(hist.history['val_loss'])
# print("===============================================")

#plot
plt.figure(figsize=(9, 5))
plt.plot(hist.history['loss'])
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

# r2스코어 :  0.607650877193654


# key-value값의 dictionary형태