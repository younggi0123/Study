import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array( [1,2,3,4,5] )
y = np.array( [1,2,3,4,5] )

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
model.summary()         # Param => 1*3+3, 3*2+2, 2*1+1

print(model.weights)    # 모델의 weight 값이 포함되어 있다
                        # kernel => weight (kernel_initializer => 가중치초기화)
                        # 훈련시키기 위해 균등분포 등으로 맞춰 잘 맞는 초기값으로 주는 initializer
print("=====================================")
print(model.trainable_weights)      # 훈련가능한 weight
print("=====================================")

print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 6
                                    # layer 1당 w, b 하나씩=> 2개  *  3layer


model.trainable=False
print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 0

model.summary()                     # trainable_param = 0

# 가중치갱신이 안된다 = 훈련이 안되고 있다

model.compile(loss='mse', optimizer='adam')


