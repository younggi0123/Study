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
le = LabelEncoder()
label = x['type']
le.fit(label)
x['type'] = le.transform(label)#transform을 이용해서 수치형 데이터를 얻어낸다(연속형)

# print(x.shape) #(3231, 12)

# 유의성 검정
#데이터를 탐색하기 위해 describe 요약통계를 확인해본다.
# print(train.describe())
# value_counts, sort_values, describe 등 요약 메서드 및 apply (darrengwon.tistory.com/427)
#value-counts(주의:value_counts()는 Series객체에서만 이용가능 하므로 반드시 DataFrame을 단일 컬럼으로 입력하여 Series로 변환한 뒤 호출한다)
# print(train['quality'].value_counts())
# 종류별 와인의 퀄리티 정보 확인
# print(train.groupby('type')['quality'].describe())
# 시리즈별 구별
# print(train.groupby('type')['quality'].quantile( [0.25, 0.5, 0.75] ).unstack("type") )
# 가장 중요한 요소인 그룹별 집합 함수에 대한 결과 확인(표준편차, 평균)
# print( train.groupby('type')['quality'].aggregate(["std", "mean"]) )
#t-검정 ㄱㄱ으로 평균에 대한 차이가 있는지 본다
import statsmodels.api as sm
red_q = train.loc[train["type"]=="red", "quality"]
white_q = train.loc[train["type"]=="white", "quality"]
# print( sm.stats.ttest_ind(red_q, white_q)[1] <= 0.05 ) # True 귀무가설 기각: "레드와인과 화이트 와인의 품질에 차이가 존재한다."
# print(  train.corr()[["quality"]]  )    # 하나니까 대괄호를 하나 더 붙이면 데이터 프레임내 결과로써 볼수 있다.(관리 용이)
train_corr = train.corr()
print(  train_corr.loc[ (np.abs(train_corr["quality"]) >0.3) & (np.abs(train_corr["quality"] != 1)), "quality" ]  )   #바로 위 코드에서 다보는게 아닌 변수를 넣어서 볼것만 본다
                                                                 # quality 가 series로 나옴(원하면 데이터프레임형태로 바꿔보셈)
                                                                 # np.abs하면 양이든 음이든 절대값으로 값이 가장 큰게 나옴
                                                                 # 1에 해당하는 부분은 제거한다 #그러므로 가장 영향이 큰 것만 남는다
                                                                 # so, 상관관계가 가장 큰 요소는 '밀도', '알코올 도수'이다.



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
scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
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
model.add(Dense(8, activation='relu'))
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

# 고정값 : 0.8trainsize, 50patience, 500epochs, default_batchs, 0.2val
# layers
# 55, activation='relu'
# 40, activation='relu'
# 30, activation='relu'
# 23, activation='relu'
# 10, activation='relu'
# 7, activation='relu'
# 5, activation='softmax'

# [MinMax] loss :  1.0041632652282715 accuracy :  0.5765069723129272
# [Standard] loss :  0.9980771541595459 accuracy :  0.599690854549408
# [Robust] loss :  0.9872334599494934 accuracy :  0.6058732867240906  ⓥcheck!ⓥ
# [MaxAbs] loss :  0.9735643267631531 accuracy :  0.5857805013656616
