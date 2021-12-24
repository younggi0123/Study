# 참고 https://daje0601.tistory.com/196?category=907960
# 참고★ https://untitledtblog.tistory.com/158
# 과적합과 Validation Dataset의 개념
#학습 규제 전략 (Regularization Strategies)

# 1. Early Stopping
#  1) 가중치가 최고 유용성 시점을 훨씬 지나서 더 업데이트 되지않도록 학습을 조기 중단함

# 모델생성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# 케라스에서는 Sequential 모델은 레이어를 선형으로 연결하여 구성합니다
model = Sequential()

# 선형으로 들어가야하기 때문에 28*28 = 784 특성 벡터로 펼쳐 변환해 Dense 층으로 들어갑니다
# # input_shape를 알려주고, 선형으로 만들어주는 Flatten을 사용합니다.
# model.add(Flatten(input_shape=(28, 28)))
# 
# # 추가적으로 Dense층을 더해줄 수 있으며, 활성화함수를 추가할 수 있다.
# #softmax는 다른 뉴런의 출력값과 상대적인 비교를 통해 최종 출력값이 결정된다.
# # 예를 들어 1,2,3이 출력이 된다면, 0.33, 0.33, 0.33이런식으로 출력이 된다는 의미있다.
# # 이에 대한 설명은 코드 하단에 추가적으로 기재해놓았으니 참고 바랍니다.
# model.add(Dense(10, activation='softmax'))
# 
# EarlyStopping early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

# 위에서 층을 설정해주었고, 최적화, loss_function, metrics를 설정하여 모델을 완성한다.
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])

# 총 7850 parameters (10 bias)
model.summary()

# 2) Parameter
'''
  -. monitor
    * val_loss 또는 val_acc로 설정이 가능하고 평가하고자 하는 대상을 출력해준다. 

  -. min_delta
    * (전 - 후) 값이 숫자보다 작으면 조기종료 발생된다. 
    * min_delta를 0 으로 설정하게 되면 살짝이라도 관찰하는 대상이 위로 튀게 되면 바로 조기 종료된다. 
    * min_delta < 0 으로 설정하면 이러한 현상으로 정지는 방지하게 된다.

  -. patience
    * 훈련이 중단 된 후 개선되지 않은 Epoch 수입니다.

  -. verbose
    * 학습 중 출력되는 문구를 설정합니다.
    * verbose=0 : 아무 것도 출력하지 않습니다.
    * verbose=1 : 훈련의 진행도를 보여주는 진행 막대를 보여줍니다.
    * verbose=2 : 미니 배치마다 손실 정보를 출력합니다.
'''

# 2. Weight Decay(가중치 감소)
# 3. Dropout
# 4. Constraint(가중치 제약)

