# OneHotEncoding의 이해
# (참고 : https://dlearner.tistory.com/22?category=828987)
# (참고2: https://wikidocs.net/22647)

import numpy as np

# 1. 데이터
from sklearn.datasets import load_wine
datasets = load_wine()
x= datasets.data
y= datasets.target
print(np.unique(y))             # [0,1,2]

# 방법은 여러가지이므로 상단 참고로 태그한 블로그와 달리 tensorflowAPI인 to_categorical을 이용하였다.
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)      #(178, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)             # (142, 13) (142, 3)
print(x_test.shape, y_test.shape)               # (36, 13) (36, 3)                

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import validation

model = Sequential()
model.add(Dense(15, activation='linear', input_dim=13))         #input dim 13될 것
#model.add(Dense(15, activation='sigmoid'))
model.add(Dense(10, activation='linear'))
model.add(Dense(7, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(11))
model.add(Dense(7, activation='linear'))
model.add(Dense(3, activation='softmax'))                      # 마지막 activation은 3이며 softmax이다

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

model.fit(x_train, y_train, epochs=100, batch_size=3, verbose=1,
          validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

# 결과값
# loss :  0.373607337474823
# accuracy :  0.9166666865348816



# (참고:https://azanewta.tistory.com/46)
# Label Encoding vs. One-Hot Encoding?그럼 어떻게 선택해 ?? 
# 사실 원론적으로 어떤 데이터셋을 쓸것인가.. 그리고 어떤 모델 기법을 적용하고 싶은가에 달렸지만,
# 몇가지 선택에 기준이 있다. 
# One-Hot Encoding은 언제?
# 순서가 없을 때 (예, 국가명 )
# 그리고 고유값의 개수가 많지 않으면 효율적
# Label Encoding은 언제?
# 순서의 의미가 있을때 (유치원, 초등학교, 대학교 같은 등급, 사원, 대리, 과장, 부장 같은 직급?? )
# 고유값의 개수가 많은데 One-hot Encoding은 당연히 메모리 소비가 많으므로 효율적이진 못하다. 
