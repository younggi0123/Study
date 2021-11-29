# ctrl+/ = 한 줄 주석처리
# import tensorflow as tf
from tensorflow.keras.models import Sequential #Sequential에 있는 놈을 끌어오겠다 # from 을 통해 tensorflow의 keras의 models의 순차적 모델을 끌어올거야
from tensorflow.keras.layers import Dense
import numpy as np



# 1. 데이터 ( 정제된 데이터 ! )
x = np.array([1.2,3])
y = np.array([1.2,3])



# 2. 모델구성
model = Sequential()                # 순차적 모델로 준비함
model.add(Dense(1, input_dim=1))    # (딱 한 단 짜리) 아웃풋, 인풋 각 하나씩_ 데이터 셋에 하나 넣어 하나 나오게 하겠다



# 3. 컴파일,  훈련
model.compile(loss='mse', optimizer='adam') # mse로 최소의 loss값을 찾겠다. mean squad error 작으면 good / loss를 부드럽게 만드는 optimizer
model.fit(x, y, epochs=5000, batch_size=1)   # epochs로 100번의 선을 그어 훈련시키겠다. batch_size는 x와y를 한번에 다 집어 넣는게 아닌, 한게씩 집어 넣는게 훈련 잘 됨(1개씩 넣겠단 말) #batch가 작을수록 훈련이 잘 되지만 표본이 너무 많으면 힘들 수 있다(속도)
# #3이 끝나면 weight와 loss값은 model에 저장된 상태이며 이를 확인하기위해 #4에서 evaluate를 사용한다


# 4. 평가, 예측
loss = model.evaluate(x, y)         #x, y로 평가하겠다
print('loss : ', loss)
result = model.predict([4])         # 4는 새로운 x값
print('4의 예측값 : ', result)      # 4로 예측한 결과

#훈련 결과를 보면 계속해서 선을 그으며 값이 4로 점점 줄어드는 것을 알 수 있으며, 훈련할때마다 값이 달라지므로 최적의 weight는 제때 저장해야 한다
