import numpy as np


# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer에 learning rate가 제시 된다. 통상 디폴트는 0.001이다.(알아서 찾아보기)
learning_rate=0.00004
# optimizer = Adam(learning_rate=learning_rate)
    # lr 0.1일때 : loss :  2.814857006072998 결과물 :  [[10.590731]]
    # lr 0.0001일때 loss :  2.531681776046753 결과물 :  [[11.055437]]
# optimizer = Adadelta(learning_rate=learning_rate)
# optimizer = Adagrad(learning_rate=learning_rate)
# optimizer = Adamax(learning_rate=learning_rate)
# optimizer = RMSprop(learning_rate=learning_rate)
# optimizer = SGD(learning_rate=learning_rate)
optimizer = Nadam(learning_rate=learning_rate)



# model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer)
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
y_predict = model.predict([11])


# '11'을 예측하고자 함
print('loss : ', round(loss,4), 'lr :', learning_rate, '결과물 : ', y_predict)



# 11 예측
optimizer = Adam(learning_rate=learning_rate)
    # lr 0.1일때 : loss :  2.814857006072998 결과물 :  [[10.590731]]
    # lr 0.0001일때 loss :  2.531681776046753 결과물 :  [[11.055437]]
optimizer = Adadelta(learning_rate=learning_rate)
    # lr 0.05일때 : loss :  2.3308 lr : 0.05 결과물 :  [[10.948152]]
optimizer = Adagrad(learning_rate=learning_rate)
    # lr 0.0001일때 : loss :  2.561 lr : 0.0001 결과물 :  [[11.2824335]]
optimizer = Adamax(learning_rate=learning_rate)
    # lr 0.00004일때 : loss :  2.4571 lr : 4e-05 결과물 :  [[10.980755]]
optimizer = RMSprop(learning_rate=learning_rate)
    # lr 0.00004일때 : loss :  2.3779 lr : 4e-05 결과물 :  [[11.265887]]
optimizer = SGD(learning_rate=learning_rate)
    # lr 0.0001일때 : loss :  2.3961 lr : 0.0001 결과물 :  [[11.282574]]
optimizer = Nadam(learning_rate=learning_rate)
    # lr 0.00004일때 : loss :  2.3747 lr : 4e-05 결과물 :  [[11.334733]]




# lr 크게 줬다가 하단에서 좌우로 핑퐁할때 작게 주면 loss가 낮은지점을 찾기가 쉬울테다.
# 100 epoch를줬는데도 성능향상이 되지않으면 stop하라.도 가능할까?!
# =>