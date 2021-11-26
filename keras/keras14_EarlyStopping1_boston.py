#  ☆★ 이번 소스를 이해하는게 중요하다!!★☆

# plot을 이용하여 validation_loss값, loss값을 체크해 본다
# 통상 val_loss가 loss보다 높고 튕긴다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import validation
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import time
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
# Data load
datasets = load_boston()
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
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping    #import한 후에
es = EarlyStopping  
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)  #정의해줘야 사용 가능
                                                                            # 조건을 달아준다(모니터할거야, 내 인내심은 20번이야, mod는 min이고 난 1번 유형으로 결과값 보겠어)
                                                                            # 최소값이 20번동안 갱신되지 않으면 멈추겠어
# Compile
model.compile(loss='mse', optimizer='adam')
start = time.time()
# History <- Fit
hist = model.fit(x_train, y_train, epochs=10000, batch_size=1,
                 validation_split=0.2, callbacks=[es] ) #모델에서 훈련동안에 E.S.를 호출할건데 es의 모니터는 val_loss로 모니터 할거다
end = time.time() - start

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