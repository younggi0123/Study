#
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Conv1D, LSTM
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.datasets import mnist

model  =  Sequential() 
model.add(Conv2D(10, kernel_size=(2,2), strides=1, padding='same', input_shape=(28, 28, 1 ) ))
model.add(MaxPooling2D())
model.add(Conv2D(5, (2,2), activation="relu") ) # Conv2D => (13,13,5)
# model.add(Dropout(0.2))
model.add(Conv2D(7, (2,2), activation="relu") ) # Conv2D => (12, 12, 7)
model.add(Conv2D(7, (2,2), activation="relu") ) # Conv2D => (11, 11, 7)
model.add(Conv2D(10, (2,2), activation="relu")) # Conv2D => (10, 10, 10)
model.add(Flatten())                            # (None, 1000)          # ← 얘는 생략 가능한 애
model.add(Reshape((100, 10)))                   # (None, 100, 10)
                                                # 괄호를 2개 치는 이유는 target_shape를 생략했기 때문이다. 원래는 요럼=> model.add(Reshape(target_shape=(100, 10)))                     # (None, 100, 10)
model.add(Conv1D(5, 2))                         # (None, 99, 5)
model.add(LSTM(15))
model.add(Dense(10, activation="softmax"))




# model.add(Dense(32))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dense(5, activation='softmax'))
model.summary()

