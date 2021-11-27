# 참고 블로그 : https://m.blog.naver.com/cjh226/221468928164

# 조기종료(early stopping)은 Neural Network가 과적합을 회피하도록 만드는 정칙화(regularization) 기법 중 하나.
# 훈련 데이터와는 별도로 검증 데이터(validation data)를 준비하고,
# 매 epoch 마다 검증 데이터에 대한 오류(validation loss)를 측정하여 모델의 훈련 종료를 제어한다.
# 구체적으로, 과적합이 발생하기 전 까지 training loss와 validaion loss 둘다 감소하지만,
# 과적합이 일어나면 training loss는 감소하는 반면에 validation loss는 증가한다.
# 그래서 early stopping은 validation loss가 증가하는 시점에서 훈련을 멈추도록 조종한다.

# TensorFlow 1.12에 포함된 Keras에서, EarlyStopping은 두 개의 파라미터를 입력받는다. 
# monitor는 어떤 값을 기준으로 하여 훈련 종료를 결정할 것인지를 입력받고, 
# patience는 기준되는 값이 연속으로 몇 번 이상 향상되지 않을 때 종료시킬 것인지를 나타낸다. 
# 위 예제로 보면 early stopping은 validation loss를 기준으로 훈련을 제어할 것이다. 
# 이 때 validaion loss가 이전 epoch보다 증가되었다고 하여 바로 중지하지는 않고, 
# 5번 연속으로 validaion loss가 낮아지지 않는 경우에 종료하도록 설정하였다. 
# 즉, patience는 모델이 아직 더 향상될 수 있지만, 우연히 validation loss가 낮게 나와버려서 
# 훈련이 종료되버리는 상황을 피하기 위한 옐로우 카드라고 생각하면 된다.

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

# Train&test&val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=66)

# 2. 모델구성
# Model Set
model = Sequential()
# Model Add
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping    #import한 후에
es = EarlyStopping  
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

# TensorFlow 1.12에 포함된 Keras에서, EarlyStopping은 두 개의 파라미터를 입력받는다.
# monitor는 어떤 값을 기준으로 하여 훈련 종료를 결정할 것인지를 입력받고,
# patience는 기준되는 값이 연속으로 몇 번 이상 향상되지 않을 때 종료시킬 것인지를 나타낸다.
# 위 예제로 보면 early stopping은 validation loss를 기준으로 훈련을 제어할 것이다.
# 이 때 validaion loss가 이전 epoch보다 증가되었다고 하여 바로 중지하지는 않고,
# 5번 연속으로 validaion loss가 낮아지지 않는 경우에 종료하도록 설정하였다.
# 즉, patience는 모델이 아직 더 향상될 수 있지만, 우연히 validation loss가 낮게 나와버려서
# 훈련이 종료되버리는 상황을 피하기 위한 옐로우 카드라고 생각하면 된다.



# Compile
model.compile(loss='mse', optimizer='adam')
start = time.time()
# History <- Fit
hist = model.fit(x_train, y_train, epochs=10000, batch_size=1,
                 validation_split=0.2, callbacks=[es] )
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( x_test )
#print("예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict)
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



# Early stopping 사용시 주의사항

# 예제에서 살펴보았듯이 TensorFlow 1.12에 포함된 Keras에서 early stopping 적용은 굉장히 쉽다.
# 하지만 patience가 0이 아닌 경우 주의해야할 사항이 있다. 위 예제를 상기해보자.
# 만약 20번째 epoch까지는 validaion loss가 감소하다가 21번째부턴 계속해서 증가한다고 가정해보자.
# patience를 5로 설정하였기 때문에 모델의 훈련은 25번째 epoch에서 종료할 것이다.
# 그렇다면 훈련이 종료되었을 때 이 모델의 성능은 20번째와 25번째에서 관측된 성능 중에서 어느 쪽과 일치할까?
# 안타깝게도 20번째가 아닌 25번째의 성능을 지니고 있다.
# 위 예제에서 적용된 early stopping은 훈련을 언제 종료시킬지를 결정할 뿐이고,
# Best 성능을 갖는 모델을 저장하지는 않는다.
# 따라서 early stopping과 함께 모델을 저장하는 callback 함수를 반드시 활용해야만 한다.



# Reference
# [1] Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.