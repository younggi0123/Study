# 앞뒤가 똑같은 오토인코더~
# 원본이 그대로 원본의 shape형태로 나온다
# (예를들면 mnist 등에서 x를 x로 훈련시키겠다.)

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations     # python에 있는 케라스도 똑같은 것임

# 1. 데이터
( x_train, _ ), ( x_test, _ ) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ),
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

# # 위아래는 같은 모델레이어임!!!!!!!!!!!!!!!

# def autoencoder(hidden_layer_size):
#     model = Sequential([
    
#     Dense(units=hidden_layer_size, input_shape=(784, ), activation='relu'),
#     Dense(units=784, activation='sigmoid')
    
#     ])
#     return model

model =autoencoder(hidden_layer_size=32)


# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, x_train, epochs=20)

output = model.predict(x_test)

# 4. 평가, 예츩
output = model.predict(x_test)

# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
        plt.subplots(2, 5, figsize=(20, 7))
        
# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


# sequential 안 레이어 형태로




# 과제 : Dense Conv2d LSTM parameter 정리하여 12시까지 보낼것 !


