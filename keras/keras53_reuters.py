from numpy.lib import index_tricks
from tensorflow.keras.datasets import reuters # 로이터통신 데이터#다항분류
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 1000, test_split = 0.2
    # 1000개가 단어사전의 개수이며 imnput dim에 들어간다
)

print((x_train), len(x_train), len(x_test))         # 8982, 2246 
print(y_train[0])                                   # 3
print(np.unique(y_train))

print( type(x_train), type(y_train) )               # < class 'numpy.ndarray'> <class 'numpy.ndarray' >
print(x_train.shape, y_train.shape)                 # (8982,) (8982,)

print( len(x_train[0]), len(x_train[1]) )           # 87, 56
print( type(x_train[0]), type(x_train[1]))          # <class 'list'> <class 'list'>

# print("뉴스기사의 최대 길이 : ", max(len(x_train)))   # error

print("뉴스기사의 최대 길이 : ", max(len(i) for i in x_train ))   # 2376 개
print("뉴스기사의 평균길이 : ", sum( map(len, x_train))/len(x_train) )  # 145.53
# map(func, *iterables) --> map object
# 데이터의 조건에 맞는애들을 출력해줌 (8982개 전체 길이(len의8982개)를 빼는데 for문 안 쓰고 쉽게가능)
# so, 다 더한애들을 전체개수로 나누면 평균

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 최대값만큼만 하고 나머진 0으로 채울거니까 2376개만큼할걸
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')   # 100보다 크면 truncate로 앞에거 자르고 뒤에거 쓰려고
print(x_train.shape)                              # (8982, 2376) => (8982, 100)

x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')   # 100보다 크면 truncate로 앞에거 자르고 뒤에거 쓰려고
print(x_test.shape)                               # (2246, 100)


print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
#총 46개


#실제 데이터 받으면 수치화는 내가 해야 하는 부분이다.


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape)     # (8982, 100) (8982, 46)
# print(x_test.shape, y_test.shape)       # (2246, 100) (2246, 46)


##################################################################################################
#  keras 53_ reuters
##################################################################################################

# word_to_index = reuters.get_word_index()
# # print(word_to_index)
# # key_value 형태의 수치화된 형태로 찍힌다.

# # 정렬안되어있으니까 key-value형태의 딕셔너리에 대한 sort만 알면 됨
# # print(sorted(word_to_index.items()))

# import operator
# # print( sorted( word_to_index().items(),key=operator.itemgetter(0) ) )   #key parameter 줘야함
# # key-value는 item getter 딕셔너리의 0번째.
# # ☆★ 딕셔너리는 키벨류쌍,딕셔너리는 키벨류쌍,딕셔너리는 키벨류쌍,딕셔너리는 키벨류쌍,딕셔너리는 키벨류쌍
# # 그렇기에 value 는 1일것이다.
# # print( sorted( word_to_index.items(), key=operator.itemgetter(0) ) )
# # print( sorted( word_to_index.items(), key=operator.itemgetter(1) ) )


# #  0 번째는 수치였는데 원래의 문자형태로 나온다.
# index_to_word= {}
# for key, value in word_to_index.items():
#     index_to_word[value+3] = key

# for index, token in enumerate(("<pad>","<sos>","<unk>")): # 요 3애들은 문장 앞부분에 붙는 부분임
#     index_to_word[index] = token
    
# print( ' '.join( [index_to_word[index] for index in x_train[0]] ) )




#2. 모델 구성 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=11, input_length=100))
model.add(LSTM(532))
model.add(Dense(346, activation='relu'))
model.add(Dense(246, activation='relu'))
model.add(Dense(146))
model.add(Dense(46, activation='softmax'))
model.summary()

#3. 컴파일, 훈련 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1000)

#평가 예측 
acc = model.evaluate(x_test, y_test)[1] # choose => y
print("acc : ", acc)