# ☆★ 참 고 ★☆
# ☆★https://circle-square.tistory.com/108★☆
# 49-2 copy
# 모델링 구성 부분

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image as keras_image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation


import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    fill_mode = 'nearest'
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])                 # 60000
print(randidx)                          # [19388 40444  5836 ... 51885 13813  7103]
print(np.min(randidx), np.max(randidx)) # 1 59998

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)                # (40000, 28, 28)   # 40000개 들어가고, shape는 28, 28이 들어가겠다.

print(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2]) # 각 40000    28     28

# (?, ?, ?) 에서 (?, ?, ?, ?)로.
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


print(x_augmented)                          # [[[[0].................]]]
print(x_augmented.shape)                    # (40000, 28, 28, 1)



x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size = augment_size, shuffle=False,
                                 ).next()[0]

print(x_augmented)                          # [[[[0.0000000e+00]........]]]]
print(x_augmented.shape)                    # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))      #(100000, 28, 28, 1)
y_train = np.concatenate((y_train, y_augmented))

print(x_train)
print(x_train.shape, y_train.shape)

# 과제
# 1. x_augment 10개와 x_train 10개를 비교하는 이미지를 출력할 것(즉, 변환 전 후 비교 )(위 10개 아래 10개로)(순서는 randidx순서)
# 물론 x_augmented부분에서 수정이 있어야겠지?
# subplot(2,10,? ) 사용      # nrows=2, ncols=10, index=?

# print(x_augmented)


print(type(x_augmented)) # <class 'numpy.ndarray'>

import matplotlib.pyplot as plt
plt.figure(figsize=(2,10))
plt.title('변환 전 후 비교')
for i in range(20):
    add_sub
    # plt.subplot(2, 10, i+1, constrained_layout = True )
    plt.axis('off')
    plt.imshow(x_train[0][i], cmap='gray')
    plt.imshow(x_augmented[0][i], cmap='gray')

plt.show()