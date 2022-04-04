# ☆★☆★☆★☆★☆★☆★☆★☆★☆★오토인코더의 강력한 기능 : 노이즈제거☆★☆★☆★☆★☆★☆★☆★☆★☆★
# 노이즈를 제거하려면..? "노이즈가 존재해야."


import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

# 0~1사이 값에 0~ 0.1 사이의 값으로 랜덤하게 더해준다 ( 본 값은 알아서 무리 되지 않는선에서 노이즈를 10% 정도로 선정한 것임 )
# 그렇다면 범위는 0~1.1 사이가 되어버리고 1을 넘는다( 최대값을 넘어버리는 값 도출 )
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

# np.clip=> 최소값은 무조건 0으로잡고 최대값은 무조건 1로잡아서 1을 넘는건 1로 퉁치겠다.
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)          # PCA에서의 95% 자리 => 154 손실율 5프로뿐

model.compile(optimizer='adam', loss='mse')


# 노이즈 : 원본  =  x  : y 번갈아가며
model.fit(x_train_noised, x_train, epochs=10)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15),
      (ax6, ax7, ax8, ax9, ax10)) = \
        plt.subplots(3, 5, figsize=(20,7))  # subplot으로 쳐서 안된건데 subplots임
        
# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


# 이는 히든레이어 들어가며 차원 축소 후 원복하는방법인데 (사실 PCA로 그냥 해도 되니까 PCA도 노이즈 제거가 가능한 부분이다.)



