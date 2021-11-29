# Early stopping이 주어진 상태에서
# patient주변의 최소값에서 멈춘건지, 최소값에서 n번 지나서 멈춘건지 판단
# (참고:https://blog.naver.com/cjh226/221468928164)
# (참고:https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
#      └(참고:https://github.com/keras-team/keras/blob/v2.7.0/keras/callbacks.py#L1710-L1855)
# (참고:https://wikidocs.net/28147)
# (참고:https://deep-deep-deep.tistory.com/1)
'''
기본적으로 patient가 끝날때까지 돈다.

early stopping은 훈련을 언제 종료시킬지를 결정할 뿐이고,
est 성능을 갖는 모델을 저장하지는 않는다. 따라서 early stopping과 함께 모델을 저장하는 callback 함수를 반드시 활용해야만 한다.
callback 함수 : restore_best_weights를 통해서 최적값에서 끝내줌

restore_best_weights

 Args:
    monitor: Quantity to be monitored.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: verbosity mode.
    mode: One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used. An epoch will be restored regardless
        of the performance relative to the `baseline`. If no epoch
        improves on `baseline`, training will run for `patience`
        epochs and restore weights from the best epoch in that set.

'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import validation
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston,load_diabetes
import time
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
# Data load
datasets = load_diabetes()
# Train_set
x = datasets.data
y = datasets.target

# Train&test&val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=66)

# 2. 모델구성
# Model Set
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping    #import한 후에
es = EarlyStopping  
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)  # 정의해줘야 사용 가능
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
