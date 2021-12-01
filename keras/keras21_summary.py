# 신경망 layer계층에서 => linear: 다 사용 가능(default임) / sigmoid: 마지막도 중간층도 다 가능 / softmax: 마지막에만 가능

# 【 relu 설명 】
# relu 넣었을때 좋아지는지 판단해본다. 성능좋아서 넣는다고 무조건 다 좋아지는 것도 아니니 테스트 해볼 것.

'''
Q) Parameter 개수?!
【파라미터의 이해】
→ 데이터 양에 비하여 너무많은 summary를 가진경우가 있을 수있기에 parameter를 보기위한 summary를 찍어보는 것이다.

예시)
(참고:https://ltlkodae.tistory.com/14)
이런 레이어가 있다고 하자/
       ㅇ
   ㅇㅇㅇㅇㅇ
     ㅇㅇㅇ
    ㅇㅇㅇㅇ
      ㅇㅇ
       ㅇ
parameter가 우리 생각 같아선 5+15+12+8+2로 42라고 생각할 것이다.
하지만 model.summary()로 찍어보면 결과값이 57이 나온다.. 아니 왜지??
왜냐면 우리는 bias란 상수값을 간과하고 선뻗은것만 생각해서 상수값은 빼고 생각했기 때문이다
y=wx에 bias도 생각해서 parameter값을 계산해야 한다.
모든 연산엔 bias의 연산까지 들어가 줘야한다.
parameter의 연산할때 summary로. 연산량을 계산해 볼 수 있다.
나중에 계산량 왕많이 늘어날때 대비해서 parameter보는법을 잘 알아둬야 한다.
나중에 CNN, RNN할 때 확실히 찾아내야 하는 부분이기에 parameter보는법을 알아두어야 한다.!
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터 
x = np.array([1.2,3])
y = np.array([1.2,3])

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))


# 데이터 양에 비하여 너무많은 summary를 가진경우가 있을 수있기에 summary를 찍어보는 것이다.
model.summary()


# 3. 컴파일,  훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=32)


# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)


#loss :  0.006493297405540943
#4의 예측값 :  [[3.859566]]


'''
                        model.summary()

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3
=================================================================
Total params: 57
Trainable params: 57
Non-trainable params: 0

 ↓ Non-trainable params는 나중에 쓰이게 된다 ↓

【 Definition of Non-trainable params 】

(https://stackoverflow.com/questions/47312219/what-is-the-definition-of-a-non-trainable-parameter)

In keras, non-trainable parameters (as shown in model.summary()) means the number of weights that are not updated during training with backpropagation.

There are mainly two types of non-trainable weights:

The ones that you have chosen to keep constant when training. This means that keras won't update these weights during training at all.
The ones that work like statistics in BatchNormalization layers. They're updated with mean and variance, but they're not "trained with backpropagation".
Weights are the values inside the network that perform the operations and can be adjusted to result in what we want. The backpropagation algorithm changes the weights towards a lower error at the end.

By default, all weights in a keras model are trainable.

When you create layers, internally it creates its own weights and they're trainable. (The backpropagation algorithm will update these weights)

When you make them untrainable, the algorithm will not update these weights anymore. This is useful, for instance, when you want a convolutional layer with a specific filter, like a Sobel filter, for instance. You don't want the training to change this operation, so these weights/filters should be kept constant.

There is a lot of other reasons why you might want to make weights untrainable.

'''