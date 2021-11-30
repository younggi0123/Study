# keras17을 사이킷런에서 제공하는 방법을 원핫인코더를 사용하여 원핫인코딩을 해본다
# https://steadiness-193.tistory.com/244
# https://ysyblog.tistory.com/71
import numpy as np

# 1. 데이터
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder

datasets = fetch_covtype()
x= datasets.data
y= datasets.target
print(np.unique(y))     # 1 2 3 4 5 7 , 7개의 라벨

# ohe.fit(data.reshape(-1,1))
# 해당 data 를 사용하여 tuple 의 형태 구축( 사실 sparse matrix 로 나옴(https://blog.naver.com/rian4u/221398406858) )
# >> 1 : (1,0,0), 2 : (0,1,0) 3 : (0,0,1) 로 매핑될 수 있게 fitting 해두는 것
# transform : 데이터 변환하기 (저장된 mapping 을 적용하는것)
ohe = OneHotEncoder(sparse=False)
y=ohe.fit_transform(y.reshape(-1,1))

print(x)
print(y)
# print(x.shape)
# print(y.shape)
# print(y.shape)  #(581012,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) #(464809, 54) (464809, 8)
print(x_test.shape, y_test.shape)   #(116203, 54) (116203, 8)
'''
# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import validation

model = Sequential()
model.add(Dense(70, activation='linear', input_dim=54))         #input_dim 54될 것
model.add(Dense(50, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(8, activation='softmax'))                      # 마지막 activation은 3이며 softmax이다

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
          validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)


# loss: 0.6848 
# accuracy: 0.6994

# loss :  0.6520121097564697
# accuracy :  0.7125031352043152

# loss :  0.6441164016723633
# accuracy :  0.7208505868911743'''