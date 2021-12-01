# keras17을 판다스에서 제공하는 get_dummies를 사용하여 원핫인코딩을 써본다
# sklearn을 쓸땐 라벨단에서 1234567인데 막상 to_categorical을 사용해서 보면 8로 찍혔었다
# \ │ 1 2 3 4 5 6 7
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 1 │ 0 1 0 0 0 0 0
# 2 │ 0 0 1 0 0 0 0
# 3 │ 0 0 0 1 0 0 0
# 4 │ 0 0 0 0 1 0 0
# 5 │ 0 0 0 0 0 1 0
# 이유인 즉, 1열의 0이 굳이 필요하지 않은데 앞에부터 채워주는 tensorflow의 to_categorical()의 경우엔 선두에 0열이 포함된다
# 예를 들어서, sex_female,sex_male의 두 열 중 하나는 필요가 없을 것이다('성별'열 은 하나만 있어도 판단 가능하니까)
#           └(https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=youji4ever&logNo=221698612004)
# 하여, 특정 원소개수만큼만 넣기 위해 sklearn의 oneHotEncoder()이나 pandas의 getdummies중 drop_first를 써준다
# 원핫인코더랑 겟더미스의 차이점은 아래 블로그를 참고해 본다.
# 더미스 설명 : https://blog.naver.com/PostView.nhn?blogId=esak97&logNo=221715216490
# 원핫인코더와 get_dummies의 차이 : https://hhhh88.tistory.com/41
#                                 https://velog.io/@jkl133/pd.getdummies%EC%99%80-sklearn.onehotencoder%EC%9D%98-%EC%B0%A8%EC%9D%B4
#                                 https://2-chae.github.io/category/1.ai/30
# └ 판다스는 사이킷런과 다르게 문자열 카테고리를 숫자 형으로 변환할 필요없이 바로 적용이 가능하다.
# └ 판다스와 넘파이는 결국 숫자까지만 지원하는 넘파이와 스트링까지 다 지원하는 판다스의 차이임
import numpy as np
import pandas as pd

# 1. 데이터
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder
# 사이킷런 데이터 에서만 .data, .target 먹히는거임!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!# 사이킷런 데이터 에서만 .data, .target 먹히는거임!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
datasets = fetch_covtype()
x= datasets.data
y= datasets.target
# print(np.unique(y))     # 1 2 3 4 5 7 , 7개의 라벨
# [컬럼지정!]
y= pd.get_dummies(y, drop_first=False)   # 열을 n-1개 생성토록 해주는 drop_first// 0으로 가득찬 열이 사라지는 걸 알 수 있다.
# 범주형 데이터 가변수/더미변수로 바꾸기 : https://zephyrus1111.tistory.com/91
# print(x)
print(y)
# print(x.shape)
# print(y.shape)
# print(y.shape)  #(581012,)
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) #(464809, 54) (464809, 8)
print(x_test.shape, y_test.shape)   #(116203, 54) (116203, 8)

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


# keras 17에 대한 간단한 정리(주관적)
# 라벨이 예를 들어 0 1 2 3 4 5 6 7 8 9이다.
# 최종적으로, 0부터 9까지를 굳이 써야한다 생각하면 tensorflow의 to_Categorical 로 간단히 0부터 나열해 9까지 쓰면 되겠다.
# (이부분은 이진분류 설명은 아니고 예를든거임) 하지만 예를 들어 2열과 3열이 성별데이터인데 한개열이 성별(남), 또 다른열이 성별(여) 라고 한다면, 굳이 성별 데이터를 두개 열거할 필요가 있을까?
# 한 개 열을 지우는게 효과적일 것이다. (ex) drop_first함수 같이 n-1열 하는 거)



# +) 이건 별개로 이진분류 이야기인데, 이진분류 역시 다중분류로 가능하다. (텐서플로우 자격증 문제에 실제로 있는 것: 말vs사람 구별문제로 이진분류 같지만, costmax를 사용하여 시그모이드가 아님)
# 남자 여자 1 2로 카테고리하면 0카테고리에서 새로운 성별로 갈리니까 안되겠지? (여자가 남자의 두배 가치를 가지지도 않는다.)
# 남자, 여자 각각 0,1로 분류하고 0+1=1이 되니 괜찮겠지
