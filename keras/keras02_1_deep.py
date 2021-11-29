# ctrl+'/' = 한 줄 # 주석처리
# ''' '''   or  """ """  범위 주석
# import tensorflow as tf
from tensorflow.keras.models import Sequential #Sequential에 있는 놈을 끌어오겠다 # from 을 통해 tensorflow의 keras의 models의 순차적 모델을 끌어올거야
from tensorflow.keras.layers import Dense
import numpy as np
#배열을 사용하기 위해서는 우선 다음과 같이 넘파이 패키지를 임포트한다. 넘파이는 np라는 이름으로 임포트하는 것이 관례이다.

# [  Q) 1,2,3,4,5를 훈련시켜서 6을 얻어보자 !   ]
# 아래 과정1 ~ 4 순서는 계속 배워 나가는동안 바뀌지 않으므로 암기한다


# 1. 데이터 ( 정제된 데이터 ! )
# 넘파이의 array 함수에 리스트를 넣으면 ndarray 클래스 객체 즉, 배열로 변환해 준다. 따라서 1 차원 배열을 만드는 방법은 다음과 같다.
x = np.array([1.2,3])
y = np.array([1.2,3])



# 2. 모델구성
model = Sequential()                # 순차적 모델로 준비함 (ex. 인공신경망)
model.add(Dense(5, input_dim=1))    # 첫 인풋만 기입하면 담부턴 생략
# ★히든레이어(https://ichi.pro/ko/hideun-leieo-lan-87226343129699)
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


# 3. 컴파일,  훈련
# 배치와 에포치란? (https://bskyvision.com/803)
model.compile(loss='mse', optimizer='adam') # mse로 최소의 loss값을 찾겠다. mean squad error 작으면 good / loss를 부드럽게 만드는 optimizer
model.fit(x, y, epochs=100, batch_size=1)   # epochs로 n번의 선을 그어 훈련시키겠다. batch_size는 x와y를 한번에 다 집어 넣는게 아닌, 한게씩 집어 넣는게 훈련 잘 됨(1개씩 넣겠단 말) #batch가 작을수록 훈련이 잘 되지만 표본이 너무 많으면 힘들 수 있다(속도)
# #3이 끝나면 weight와 loss값은 model에 저장된 상태이며 이를 확인하기위해 #4에서 evaluate를 사용한다


# 4. 평가, 예측
loss = model.evaluate(x, y)         #x, y로 평가하겠다
print('loss : ', loss)
result = model.predict([4])         # 6은 새로운 x값
print('6의 예측값 : ', result)      # 예측한 결과

#훈련 결과를 보면 계속해서 선을 그으며 값이 6으로 점점 줄어드는 것을 알 수 있으며, 훈련할때마다 값이 달라지므로 최적의 weight는 제때 저장해야 한다

'''
loss를 낮춘 값이
10번 중 최소의 loss가 6의 예측값이 된다
6의 예측값이 6인게 정답이 아닌
예를 들어 10~20번했을때 loss가 가장 작은 값이 데이타 셋에서 가장 좋은 값이 된다.
'''




# MSE(Mean Squared Error)
# https://wikidocs.net/34063
# 글로 적기 힘든 부분_ 블로그 글 읽기