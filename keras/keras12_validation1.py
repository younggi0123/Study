# 21.11.26 validation1. validation 사용 이유
# ☆★ Train&Test&Validation set을 나누는 이유? ★☆
# https://wikidocs.net/31019
# validation(검증)으로 머신이 먼저 문제를 풀어본다('훈련시킨 것'을 검증. 결과에 영향을 미치지는 않아)
# #3. fit에서 validation_data부분을명시


# ValidationSet이해
# https://wikidocs.net/31019

# train,test set을 분리하는 이유!!!!!!
# (참고:https://teddylee777.github.io/scikit-learn/train-test-split)
# (참고:https://ysyblog.tistory.com/69)


# 향후에도 평가에 있어서 train_loss가 아닌 val_loss가 중요한 경우가 많다  
# Validation loss가 Train loss보다 낮은 이유에 대한 글
# (참고:https://koreapy.tistory.com/577)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터

# Train_set
x_train = np.array(range(10))
y_train = np.array(range(10))
# Test_set
x_test  = np.array( [11, 12, 13] )
y_test  = np.array( [11, 12, 13] )
# Validation_set
x_val = np.array( [14, 15, 16] )
y_val = np.array( [14, 15, 16] )

# 2. 모델구성
# Model Set
model = Sequential()
# Model Add
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
# Fit
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))   #validation은 훈련 도중에 들어가겠지?!

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( [17] )
print("17의 예측값 : ", y_predict)

# train_loss는 훈련에 과적합되어 있기 때문에 validation_loss가 더 좋다
# validation loss가 train loss 보다 수치가 낮은 이유에 대한 글
#https://koreapy.tistory.com/577