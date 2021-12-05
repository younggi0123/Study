# dacon_wine
# 학우 여러분들 항상 감사합니다 제 코드는 여러분들 git을 열심히 참고한 콤비네이션 산유물 입니다.ㄳㄳ

# argmax함수 : max와 달리 값 자체가 아닌 색인 위치를 반환한다
#https://www.delftstack.com/ko/api/numpy/python-numpy-argmax/
# 【       Subject : 21'. 12. 03. keras24_Dacon_Wine       】


# 1. how to choose which is the best algorithm in this data set?
# 참고nexablue.tistory.com/29 → 본 데이터 셋은 다중분류 ㄱㄱ

from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from pandas import get_dummies

#1 데이터
path = "../_data/dacon/wine/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

y = train['quality']
x = train.drop(['id', 'quality'], axis =1)

#라벨인코더 사용법 다시 찾아보기
#참고steadiness-193.tistory.com/243
le = LabelEncoder()#
label = x['type']
le.fit(label)
x['type'] = le.transform(label)#transform을 이용해서 수치형 데이터를 얻어낸다

print(x.shape) #(3231, 12)

test_file = test_file.drop(['id'], axis=1)
label2 = test_file['type']
le.fit(label2)
test_file['type'] = le.transform(label2)

y = train['quality']
# print(y.unique())#[6 7 5 8 4]
y = get_dummies(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

# 전처리 4대장
# a. MinMaxScaler
# scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
scaler = RobustScaler()
# d. MaxAbsScaler
# scaler = MaxAbsScaler()
# # Scaler fit & transform
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)#테스트파일도 잊지말고~

#2 모델
model = Sequential()
model.add(Dense(55, activation='relu', input_dim = x.shape[1]))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.save("./_save/dacon_wine_quality_save_model.h5")

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs = 500, batch_size =32,validation_split=0.2,callbacks=[es])
end = time.time() - start
print('Time :', round(end,2) ,'sec')

#4 평가
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0])
print("accuracy : ",loss[1])

################# 제출용 #################
result = model.predict(test_file)
# print(result[:5])
result_recover = np.argmax(result,axis=1).reshape(-1,1)+4
# print(result_recover[:5])
# print(np.unique(result_recover))
submit_file['quality'] = result_recover

# print(submission[:10])
submit_file.to_csv(path + "wine_quality_submit.csv", index = False)
# print(result_recover)


# 결과
# [MinMax] loss :  1.0041632652282715 accuracy :  0.5765069723129272
# [Standard] loss :  0.9980771541595459 accuracy :  0.599690854549408
# [Robust] loss :  0.9758855700492859 accuracy :  0.5919629335403442
# [MaxAbs] loss :  0.9735643267631531 accuracy :  0.5857805013656616