# 1. 회귀인지 이중분류인지 다중분류인지 판단 한 후 keras15~16을 토대로 진행해 본다
# ★☆★☆★☆★☆★☆★☆★☆★☆ 회귀인지 이중분류인지 다중분류인지 판단법  =>  ????????????????????????
# ★☆★☆★☆★☆★☆★☆★☆★☆       └ (https://nexablue.tistory.com/29)
# 2. fit의 배치사이즈를 지우고 돌려본 후 결과를 적는다
#    +) 배치사이즈의 default는 무엇인가?
# batch_size: 정수 혹은 None. 경사 업데이트 별 샘플의 수. 따로 정하지 않으면 batch_size는 디폴트 값인 32가 된다.
# 32는 어떻게 알 수 있을까? 예를 들어, batch_size를 1로 설정하고 돌리면 본 source에서는 1epoch당 batchsize가 371847이 나온다.
# 다시, fit부분에서 batch_size를 제거하고 돌려보면 11621이 나온다.
# 371847을 11621로 나눠보면 31.9978...으로 batch_size의 default값은 약 32인 것을 확인할 수 있다.


# 텐서플로우에서 제공하는 to_categorical 방법을 이용하여 원핫인코딩을 써본다

import numpy as np

# 1. 데이터
from sklearn.datasets import fetch_covtype
datasets = fetch_covtype()
x= datasets.data
y= datasets.target
print(datasets.DESCR)
print(np.unique(y))     # 1 2 3 4 5 7 , 7개의 라벨
# print(x)
# # print(y)
# print(x.shape)
# print(y.shape)
# print(y.shape)  #(581012,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
# print(y.shape)

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