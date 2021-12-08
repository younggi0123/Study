import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000, ) #28,28,1과 같이 흑백이다.
print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000, ) #4차원으로 만들어줘야 reshpae해서 cnn이 가능할 것

print(x_train[0])
print('y_train[0]번째 값 : ', y_train[0])
# y값은 곧 숫자그림 5이다.
# 라벨은 총 9 10개다.

import matplotlib.pyplot as plt
plt.imshow(x_train[0], 'gray')  # 이미지에 대한 shape을 넣어주면 바로 출력되는 imshow
plt.show()