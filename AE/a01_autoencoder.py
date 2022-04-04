# EX. GAN을 하는데 시간이 너무 오래걸려요? 필요없는 애를 뺀 PCA에서 작업하는 것도 방법이겠지?

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations     # python에 있는 케라스도 똑같은 것임

# 1. 데이터
( x_train, _ ), ( x_test, _ ) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape = (784, ))

# encoded = Dense(16, activation='relu')(input_img)
# encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(154, activation='relu')(input_img)          # PCA 0.95 지점
# encoded = Dense(486, activation='relu')(input_img)            # PCA 0.999 지점
# encoded = Dense(1024, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)
# 특이점 ?! => 흐려짐(마지막 결과 보면 같은사진 같지만 그렇지 않음)

autoencoder = Model(input_img, decoded)


# 시그모이드&바이너리 크로스 엔트로피 많이 사용

# 3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

# 3. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4)) #도화지
for i in range(n):
    # x_test원본출력
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # x_
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow( decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 결과값보면 특성(특히, 가장자리)을 잡아내며 이미지가 많이 흐려졌음