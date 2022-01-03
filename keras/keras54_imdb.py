from tensorflow.keras.datasets import imdb#imdb 이진분류
import numpy as np

(x_train, y_train),(x_test,y_test) = imdb.load_data(
    num_words=10000
    # https://keras.io/ko/preprocessing/text/
    # num_words: 단어 빈도에 따른 사용할 단어 개수의 최대값. 가장 빈번하게 사용되는 num_words개의 단어만 보존합니다.
)


print(x_train.shape, x_test.shape)      #(25000,) (25000,)
print(y_train.shape, y_test.shape)      #(25000,) (25000,)
print(np.unique(y_train))               #[0 1]
print(x_train[0], y_train[0])
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 
#  1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32] 1


#@ 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=2494, padding='pre') 
x_test = pad_sequences(x_test, maxlen=2494, padding='pre')  

#@ y확인
print(np.unique(y_train)) # [0, 1]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (25000, 2), (25000, 2)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation="softmax"))

#@ 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            metrics=['acc'])
model.fit(x_train, y_train, epochs=5, batch_size=200)

#@ 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)

'''
acc :  0.8196799755096436
'''
