# 수업

# DL<ML<AI

# 튜닝-진공관-컬큘레이터~
# 시그모이드-렐루 등 활성화함수~

# 퍼셉트론이란??
# 퍼셉트론은 왜 망했어?
# 인공지능의 첫번째 암흑기
# http://scimonitors.com/ai%EA%B8%B0%ED%9A%8D%E2%91%A1-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-%EB%B0%9C%EB%8B%AC%EA%B3%BC%EC%A0%95-%ED%8A%9C%EB%A7%81%EB%B6%80%ED%84%B0-%EA%B5%AC%EA%B8%80-%EC%95%8C%ED%8C%8C%EA%B3%A0-ibm/
# Why XOR gate ?



# keras16을 머신러닝으로 리폼해 본다.

import numpy as np
from sklearn.datasets import load_iris

datasets = load_iris()

x = datasets.data
y = datasets.target

# 기본적 레거시한 머신러닝은 카테고리컬 조차 필요가 없다.
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)      #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC

# model = Sequential()
# model.add(Dense(10, activation='linear', input_dim=4))
# model.add(Dense(15, activation='sigmoid'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(7, activation='linear'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(11))
# model.add(Dense(7, activation='linear'))
# model.add(Dense(3, activation='softmax'))

model = LinearSVC() # 모델구성 얘 하나로 끝



# # 3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
# # Fit
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
#                    verbose=1, restore_best_weights=True)    
# model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
#           validation_split=0.2, callbacks=[es])

model.fit(x_train, y_train) #   훈련 이걸로 끝

# 4. 평가, 예측
# Evaluate
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss[0])
# print('accuracy : ', loss[1])

# 매트릭스 개념 어큐러시 반환(분류지표임(분류모델))
# 모델에서 판단해 주므로 평가지표가 R2여야 하면 R2로 알아서 도출해 준다.
result = model.score(x_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("result : ", result)
print("accuracy-score : ", acc)

# results = model.predict(x_test[:7])
# print(y_test[:7])
# print(results)
