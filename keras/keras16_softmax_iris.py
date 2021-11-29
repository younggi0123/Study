# softmax함수를 iris data에 적용한다

import numpy as np
from sklearn.datasets import load_iris

datasets = load_iris()
# print(datasets.DESCR)
# :Number of Instances: 150 (50 in each of three classes)                # 열이 50 개씩 3개가 있다
# :Number of Attributes: 4 numeric, predictive attributes and the class  # 4개의 컬럼이 있다
# print(datasets.feture_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)       # (150, 4 ) (150,)
# print(y)
# print(np.unique(y))           # [0 1 2]
                                # 다중분류라고 명시되지 않아도 라벨값이 세개 이상이면 다중분류 이겠거니 당연히 하는 것

# y를 다중인코딩으로 변환시키는 법 (150,4)와 (150,3)이 되어야 함. 방법은 사이킷런에 2개 텐서플로 1개
# 지금까진 알고리즘으로 했겠지만 여기선 api가져다 쓴다 아래와 같이 ㄱㄱ한다
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)      #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import validation

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=4))         #input dim 4될 것
model.add(Dense(15, activation='sigmoid'))
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
                   verbose=1, restore_best_weights=True)    

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
          validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

# 1차 결과
# loss :  0.08911363780498505
# accuracy :  0.9666666388511658
'''
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
 위와 밑을 매칭해보면 첫줄~셋째줄이 -01로 둘째 1과 매칭되고 4번째가 -01이니까 또 첫째1과 매칭됨
[[1.2313690e-04 9.9805605e-01 1.8208810e-03]
 [1.6854012e-05 9.2253214e-01 7.7451013e-02]
 [2.2420889e-05 9.3513274e-01 6.4844914e-02]
 [9.9989629e-01 1.0370419e-04 5.0098944e-23]
 [5.0743212e-05 9.9122131e-01 8.7279603e-03]
 [1.7998122e-04 9.9889791e-01 9.2218129e-04]
 [9.9993813e-01 6.1831939e-05 1.3887052e-23]]
 '''
# e의 - 숫자가 가장 작은게 큰 놈일 것이여

# Categorical Encoding의 두가지 방법

# Label Encoding
# One-Hot Encoding
'''
Label Encoding 이란 알파벳 오더순으로 숫자를 할당해주는 것을 말한다.
글자니까 당연히 알파벳순으로 정렬이 가능 할 것이고... 그 정렬된 기준으로 번호를 매긴다는 뜻이다
Country Age Salary
India   44  72000
US      34  65000
Japan   46  98000
US      35  45000
Japan   23  34000

위의 Country 열의 정보를 숫자로 바꾸면 다음과 같다
Country Age Salary
0       44  72000
2       34  65000
1       46  98000
2       35  45000
1       23  34000

하지만 Label Encoding의 적용이 쉽지만은 않은데,
위 시나리오에서 Country라는 데이터는 순서나 랭크가 없다. 그러나 Label Encoding을
수행하면 결국엔 알파벳 순으로 랭크가 되는 것이고, 그로 인해서 랭크된 숫자정보가
모델에 잘못 반영될수가 있겠다.

그렇기에 One-Hot Encoding을 통해서 카테고리를 목록화 해서 개별로 목록값에 대한 이진값으로 만드는 방법이다.

<< One-Hot Encoding은 더미변수를 만든 것을 처리하는 방법이다. >>
아래와 같다. 인디아가 100, 일본이 001, 미국이 002 이런 느낌이다.
0      1      2      Age    Salary
1      0      0      44     72000
0      1      0      46     98000
0      0      1      35     45000
0      1      0      23     34000

One-Hot Encoding은 적용에 문제가 없을까 ?? : Dummy Variable Trap
Dummy Variable Trap은 One-Hot Encoding한 변수 하나의 결과가 다른 변수의 도움으로 쉽게 예측되어 질수 있다는 것이다.  
Dummy Variable Trap 변수들이 각각 다른 변수들과 상관성이 있다는 것.
The Dummy Variable Trap은 multicollinearity. 라는 문제를 야기 시킨다.
Multicollinearity 는 독립적인 Feature간에 의존성이 있을때 발생한다.
(Multicollinearity은 선형 회귀나, 로지스틱 회귀에서 심각한 이슈가 됨.)
이걸 극복하기 위해서는 dummy variables 중 하나를 버려야 된다. 
Onehot Encoding후 이 문제를 해결하는 예시를 하나 같이 검토해보자. 
일반적으로 multicollinearity를 체크하는 방법중 하나는 Variance Inflation Factor를
활용하는 것이다.(이부분은 블로그보고 https://azanewta.tistory.com/46)
어떻게 선택해 ?? Label Encoding vs. One-Hot Encoding?
사실 원론적으로 어떤 데이터셋을 쓸것인가.. 그리고 어떤 모델 기법을 적용하고 싶은가에 달렸지만,

몇가지 선택에 기준이 있다. 
One-Hot Encoding은 언제?
순서가 없을 때 (예, 국가명 ) 그리고 고유값의 개수가 많지 않으면 효율적
Label Encoding은 언제?

순서의 의미가 있을때 (유치원, 초등학교, 대학교 같은 등급, 사원, 대리, 과장, 부장 같은 직급?? )
 고유값의 개수가 많은데 One-hot Encoding은 당연히 메모리 소비가 많으므로 효율적이진 못하다. 


'''