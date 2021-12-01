# 0. from keras14:회귀모델, keras15:이진모델(Sigmoid)

# 1. '유방암 데이터'를 load하여, 활성함수 sigmoid에 대한 이해한다
# 2. 컴파일할때 metrics(거리함수)의 이용(accuracy)하는 걸 확인한다.
# 3. 0과 1뿐인 sigmoid와 다르개 3개일때의 다중분류 - '소프트맥스 함수'를 이해한다
# 4. oneHot Encoding을 이해한다

# +) 컴파일할때 metrics(거리함수)의 이용(accuracy)
# loss = mse = binary cross entropy
# 100% 이진분류는 binary cross entropy
# ★ 이진분류의 액티베이션은 시그노이드 ★
# ★ 이진분류의 로스는 바이너리 크로스 엔트로피 ★
# 이진분류는 다중분류에 포함이다.

# 0과 1의 이진분류이기에 이진분류 모델을 사용하는 것이다(이전번까지는 다 회귀모델 사용했지만,)

# +) 앞의 소스에서 회귀모델의 보조지표로 r2를 썼는데
# 이번건 분류인데 r2가 먹힐까? 수치야 나오겠지만 쓸모 없을 것
# 하지만, loss는 통상적으로 쓰는 loss와 동일하다.
# loss가 낮은게 좋겠다, 보조평가지표는..?!

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR) # PANDAS Decribe
print(datasets.feature_names)

x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=66)
# print(x.shape, y.shape) # (569, 30) (569, )
# print(y_test[:11])                      # y를 찍어보며 이진분류인지 판단 (y의 라벨값이 어떤것이 있는지 파악, unique하니 0과 1이여. 오호 0과 1이라니.. 그렇다면 시그모이드구먼?! 그게 아니라면 BinaryCrossEntropy겠구먼!!)
                                          # y 찍어보니 0 0 0 0 0 0 0이 나온다 이 친구 시그모이드구먼..
print(np.unique(y))     # [0 1]  np.unique함수를 통해 [0,1]인 이진모델임을 확인했다  https://www.delftstack.com/ko/api/numpy/python-numpy-unique/

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import validation

model = Sequential()
model.add(Dense(20, activation='linear', input_dim=30))       #feature input 다시정리!!!
model.add(Dense(15, activation='sigmoid'))              # 여기서 sigmoid가 나와도 틀린건 아님. 왜냐면, 전 값이 크다 생각하면 0과 1사이로 한정해놓고
                                                        # 다음턴에 다시 linear할 수 있는 것이다.(늘렸다 줄이는 등 이런건 해보며 느끼는..)
                                                        # linear로 연산된걸 sigmoid로 감싸서 값을빼서 0~1사이의 값으로 나타낸다
model.add(Dense(10, activation='linear'))
model.add(Dense(7, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(11))                                    # 이처럼 activation을 특정하지 않아도 default값은 linear로 정해져 있다.
model.add(Dense(7, activation='linear'))
model.add(Dense(1, activation='sigmoid'))               # ★ 마지막은 반드시 시그모이드 ★ 마지막은 반드시 시그모이드 ★ 마지막은 반드시 시그모이드 !
                            # delete왜 마지막은 시그모이드?
# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )               # 이진분류임을 인지시킨다
                                                                                                 # metrics 부분이 대괄호인 이유는 다른 애들을 또 쓸 수도 있으니까.
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto', # loss같이 낮은게 좋다?, max? 재정리 ㄱㄱ
                   verbose=1, restore_best_weights=True)    

model.fit(x_train, y_train, epochs=100, batch_size=3, verbose=1,
          validation_split=0.2, callbacks=[es])  # 이 때, 콜백을 반환해주는건 es를 콜백으로 반환에 쓰겠다는건데, 대괄호라는건 다른 콜백이 또 있다는 소리겠지?(리스트니까)

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

results = model.predict(x_test[:11])
print(y_test[:11])
print(results)

# 1차 결과
# loss :  [0.6665129661560059, 0.6432748436927795]
# 2차 결과
# loss :  [0.5496556162834167, 0.8070175647735596]


'''

[ 결과값 출력 ]
loss [0.6567910313606262, 0.6432748436927795]
이렇게 나오는데,
evaluate했을 때 소스 compile 부분과 매치되는 부분은 다음과 같다.
左 ▶ 0.6567910313606262 ▶ loss='binary_+crossentropy'    항상 첫 번째는 loss 값 고정임!!!!!!!
右 ▶ 0.6432748436927795 ▶ metrics=['accuracy']           

metrics는 현재 상황만 출력해 주는 것 평가지표에 의한 값이 어떤 모양으로 돌아가는지.
결과값에 accuracy로 출력하여 보여만준다.

진짜 좋은 모델은 어차피 loss로 판단되어진다.
하지만 loss와 accuracy가 같은 방향으로 움직이거나 이상할때?!
loss중에서도 val_loss(검증결과 validation)를 더 판단.

accuracy가 필요 없을 것 같아도
상대지표 아닌 절대지표도 필요한 것이다.
(예를들어 내가 경시대회3등인데 잘했다! 라고 판단했지만 알고보니 1등이 50만명이 있으면 내가 잘한게 아니니까..)
내가 생각한 정확도는 99%인데 실제론 33%였다..


# ★★마지막 output_layer의 activation은 반드시 sigmoid를 사용한다 ★★
# ★★데이터를 위치로 해서 잘라버리는 평기자표를 binary_crossentropy를 쓴다.! ★★




┌                    ┐
까먹지 말라고 또 씀
- loss : 훈련 손실값
- acc : 훈련 정확도
- val_loss : 검증 손실값
- val_acc : 검증 정확도
acc는 accuracy '정확도'라는 뜻으로 값이 1에 가깝고 높을 수록 좋은 모델이라는 것을 뜻합니다.
loss는 결과 값과의 차이를 의미하므로 작을 수록 좋고 0.0000000....에 수렴할수록 좋은 모델이라는 것을 뜻합니다.
이건 유연하게 봐야하겠지만, 
epoch의 횟수가 많아질수록(훈련횟수가 많아질수록) loss는 줄어들고 accuracy는 증가하게 되고
어느 시점이 되면 일정 값에 수렴하다가 성능이 점점 떨어지기도 합니다.(오버피팅)
└                    ┘


'''




# 참고 : Sigmoid 대신 ReLU? 상황에 맞는 활성화 함수 사용하기
# https://medium.com/@kmkgabia/ml-sigmoid-%EB%8C%80%EC%8B%A0-relu-%EC%83%81%ED%99%A9%EC%97%90-%EB%A7%9E%EB%8A%94-%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-c65f620ad6fd
# https://m.blog.naver.com/zzoyou_/222014804966



# 다중분류 - 소프트 맥스 함수
# https://m.blog.naver.com/wideeyed/221021710286
# Softmax(소프트맥스)는 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수이다.
# 분류하고 싶은 클래수의 수 만큼 출력으로 구성한다.
# 가장 큰 출력 값을 부여받은 클래스가 확률이 가장 높은 것으로 이용된다.
# https://gooopy.tistory.com/53
# 소프트맥스는 세 개 이상으로 분류하는 다중 클래스 분류에서 사용되는 활성화 함수다.
# 소프트맥스 함수는 분류될 클래스가 n개라 할 때, n차원의 벡터를 입력받아, 각 클래스에 속할 확률을 추정한다.


# 원-핫 인코딩(One-Hot Encoding)이란?
# 원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식입니다. 이렇게 표현된 벡터를 원-핫 벡터(One-Hot vector)라고 합니다.

# 원-핫 인코딩을 두 가지 과정으로 정리해보겠습니다.
# (1) 각 단어에 고유한 인덱스를 부여합니다. (정수 인코딩)
# (2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여합니다.


# 4. Softmax 자체는 큰 아이를 찾기 때문에 큰쪽을 찾는데 예를들어 0, 1, 2가 주어졌을때 2가 1의 2배가 되는 격
# 무조건 2배가 되면 안되고 공평해야하며 이 때 oneHot Encoding을 사용한다