import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D

tf.compat.v1.set_random_seed(66)

# 1. 데이터
from keras.datasets import mnist
(x_train,y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float')/255.

x = tf.placeholder( tf.float32, [None, 28, 28, 1] )
y = tf.placeholder( tf.float32, [None, 10] )

# 2. 모델 구성
w1 = tf.get_variable( 'w1', shape=[3, 3, 1, 32] )# w1이름 아무렇게나 # shape 자르기 => kernel_size ( Kernel_size just like Weight )
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')
# filter = output
# !!!! stride를 2by2하려면 1,2,2,1 이런식으로 넣어줘야함(앞 뒤는 4차원을 맞추기위해 존재할 뿐)
# model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), padding='valid', input_shape=(28, 28, 1)))

# x * w1  커널사이즈 하나 줄었음 (None,27,27,64)
print(w1)
print(L1)
