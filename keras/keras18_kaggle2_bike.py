'''
# 과제 : 평균VS중위수 비교
# [평균]
# 평균은 자료에 포함된 관측치를 모두 더한 후 관측치의 수로 나눈 값을 말한다.  5명의 시험 점수가 60, 55, 65, 70, 65라고 하면 총 5로 나누어 63이 된다. 
# [중위수]
# 평균이 아닌 자료의 중심위치를 나타내는 통계량으로서 많이 쓰이는 것이 중위수이다. 중위수는 자료를 크기 순으로 정리할 경우 중간에 위치한 값을 의미한다.
# 예를 들어 20,21,22,23,24가 있다면 22가 중위수가 된다. 단, 수가 짝수일 경우 앞,뒤의 값의 평균이 중위수가 된다. 중위수는 정의상 그 이상의 값을 가지는 값들이 50%이상이고 그이하를 가지는 값이 50%가 된다. 
# [결론]
# 때로는 평균보다 중위수를 사용하여 중심위치를 파악하는 것이 바람직할 경우가 있다.
# 그 이유는 평균 경우 극단치에 큰 영향을 받기 때문이다. 그렇기에 극단치가 있다고 생각될 경우 평균과 중위수를 같이 구하여 중심위치를 파악하는게 좋다.
'''


# bike데이터의 train.csv 데이터 세트를 이용해 모델을 학습한 후 대여 횟수(count)를 예측해본다.
# Kaggle_bike_sharing_demand_data

# ★★★★★링크)https://www.kaggle.com/dogdriip/bike-sharing-demand★★★★★
# ★★★★★└꼭봐└꼭봐└꼭봐└꼭봐└꼭봐└꼭봐└꼭봐└꼭봐└꼭봐└꼭봐└꼭봐└꼭봐★★★★★

# RMSE, LRMSE, def로 함수 정의하기

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import validation
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error    # r2, mse


# ★ 캐글에서 요구한 성능 평가 방법은 RMSLE(Root Mean Square Log Error)이다. 즉, 오류 값의 로그에 대한 RMSE.
# ★ 아쉽게도 사이킷런은 RMSLE를 제공하지 않아서 RMSLE를 수행하는 성능 평가 함수를 직접 만들어야 한다.
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

# 1. 데이터
path = "./_data/titanic/"
path = "./_data/bike/"
train = pd.read_csv(path + 'train.csv')
# print(train)        # (10866, 12)
test_file = pd.read_csv(path + 'test.csv')          # test라고 지으면 애매해서
# print(test_file)         # (6493, 9)      train에서 casual,regis,count의 3개가 빠진 데이터
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
# print(submit_file)       # (6493, 2)

# print(submit_file.columns)   #['datetime','count']

# 판다스에서 사이킷런이 제공하는 툴인 .DECR와 비슷한 기능
# ★ 근데 찍어보니까 datetime이 object형으로 뜨네??? why..?!
#                                                        └  = object로 뜬다 = string(문자열)이다. ★
# 그러니 시간데이터를 쓴다면 나중에 다시 date형으로 바꿔줘서 써야 한다.
# print(train.info())             # <class 'pandas.core.frame.DataFrame'>
# print(type(train))
# print(train.describe())
# print(train.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'], dtype='object)
# print(train.head(3))
# print(train.tail())

# ★(↑Index참고)컬럼 y는 season~windspeed까지 8개로 구성하고(datetime은 object로 string타입이니까 제외함) x는 count로 구성한다.
x = train.drop( ['datetime', 'casual', 'registered', 'count'], axis=1) # 이렇게 4개 빼고 컬럼 구성
test_file = test_file.drop( ['datetime'], axis=1) # model.predict에서 돌아가게 하도록 datetime 오브젝트를 지운다.

# ★★ error!!!!  axis 오류 = 축오류 : 가로단위 아니고 세로단위로 생각하려면 axis=1을 넣어 컬럼상태로 만들어 줘야 한다. 0이면 행단위로 삭제가 됨!!!!!!!!!!!!!
# print(x.columns)
# print(x.shape)      # (10886, 8)
y = train['count']
# print(y.shape)      # (10886, )
# plt.plot(y)
# plt.show()
# 플롯을 확인해 보니까 데이터가 너무 치우쳐 왜곡되어 있었다..! 그렇다면 로그를 씌워서 성능을 좋아지게 해본다 (로그함수가 성능이 더 좋은건지는 검색ㄱㄱ;;)
# 나중에 predict할때 다시 지수변환 하면 되니까 괜찮( 지금 뭔소린지 몰라도 다 필기 중임 ㅠㅠㅠㅠ )
# 로그 변환시의 치명적 오류가 있음,, log1은 0인데 log0은 정의될수 없는 숫자이다. 그러므로 0이 들어가지 않게 주의해야한다 (0이 되지 않게 주의)
# 0이 안 되게 하려면 로그변환 하기 전에 1을 더해주면 된다 (그러면 아무리 작아도 최소값은 log1=0이니까 0임)
# 로그변환은 넘파이에서 제공한다.


'''
https://www.kaggle.com/dogdriip/bike-sharing-demand
이제 회귀 모델을 이용해 자전거 대여 횟수를 예측한다.
회귀 모델을 적용하기 전에 데이터 세트에 대해서 먼저 처리해야 할 사항이 있다.
결과값이 정규 분포로 돼 있는지 확인하는 것과 카테고리형 회귀 모델의 경우 원-핫 인코딩으로 피처를 인코딩하는 것.
회귀에서 큰 예측 오류가 발생할 경우 가장 먼저 살펴볼 것은
Target 값의 분포가 왜곡된 형태를 이루고 있는지 확인하는 것이다.
Target 값의 분포는 정규 분포 형태가 가장 좋다.
그렇지 않고 왜곡된 경우에는 회귀 예측 성능이 저하되는 경우가 발생하기 쉽다.
count 칼럼 값이 정규 분포가 아닌 왜곡돼 있는 것을 알 수 있다.
이렇게 왜곡된 값을 정규 분포 형태로 바꾸는 가장 일반적인 방법은 로그를 적용해 변환하는 것.
여기서는 넘파이의 log1p()를 이용하겠다. 이렇게 변경된 Target 값을 기반으로 학습하고
예측한 값은 다시 expm1() 함수를 적용해 원래 scale 값으로 원상 복구하면 됩니다.
log1p()를 적용한 count값의 분포를 확인한다.
'''


# ★★rmsle() 함수를 만들 때 한 가지 주의해야 할 점
# rmsle를 구할 때 넘파이의 log() 함수를 이용하거나 사이킷런의 mean_squared_log_error()를 이용할 수도 있지만
# 데이터 값의 크기에 따라 오버플로/언더플로 오류가 발생하기 쉽다.
# 따라서 log()보다는 log1p()를 이용하는데, log1p()의 경우는 1+log() 값으로 log 변환값에 1을 더하므로 이런 문제를 해결해 준다.
# 그리고 log1p()로 변환된 값은 다시 넘파이의 expm1() 함수로 쉽게 원래의 스케일로 복원될 수 있다.
#   ↓
# 로그변환
y = np.log1p(y)

# Train&test&val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.3)#, shuffle=True, random_state=66

# 회귀식이라도 카테고리형 변수를 one-HotEncoding을 해준다.
# 사이킷런은 카테고리를 위한 데이터 타입이 없으며, 모두 숫자로 변환해야 한다.
# 하지만 이처럼 숫자형 카테고리 값을 선형 회귀에 사용할 경우
# 회귀 계수를 연산할 때 이 숫자형 값에 크게 영향을 받는 경우가 발생할 수 있다.
# 따라서 선형 회귀에서는 이러한 피처 인코딩에 원-핫 인코딩을 적용해 변환해야 한다.
# PANDAS의 get_dummies()를 활용, 이러한 영향을 미치는 칼럼을 비롯한 나머지 칼럼도 전부 원핫인코딩 하고 다시 성능예측해 본다.
#train = pd.get_dummies(train, columns=['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'])#블로그보고 추가함
#test_file = pd.get_dummies(test_file, columns=['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'])#블로그보고 추가함
# ☆되나 안되나 찍어보는 습관
# print(train.shape)          #(10886, 242)
# print(test_file.shape)      #(6493, 232)

# train_df와 test_df의 shape를 맞춰주기 위해 align을 사용한다. (https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding )
# train, test_file = train.align(test_file, join='left', axis=1)#블로그보고추가함
# test_df = test_file.drop(['count'], axis=1)#블로그보고추가함
# print(train.shape)          #(10886, 242)
# print(test_file.shape)      #(6493, 242)
#그리고 다시 train을 set해본다.
#x_train, x_test, y_train, y_test = train_test_split(train.drop(['count'], axis=1), train['count'], test_size=0.3)#블로그보고 추가함

# 2. 모델링 구성
# Model Set
model = Sequential()
model.add(Dense(6, input_dim=8))
model.add(Dense(8, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(1))

# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test)

# Predict
r2 = r2_score(y_test, y_pred)
print('r2스코어 : ', r2)

# RMSLE와 비슷한 효과의 rmse이다.
rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)

# 결과값
# 로그변환 전
# loss : 23728.705078125
# R2 : 0.24927524076414598      # r2 0.5이하면 대체로 쓰레귀..
# RMSE : 154.04125011447778

# 로그변환 후
# loss :  1.4505070447921753
# r2스코어 :  0.2597771330436215
# RMSE :  1.2043699649901036

# ★☆★☆★☆★☆ predict한 결과는 submit변수의 count에 넣은 후 csv로 넣어주면 되겠다.! ★☆★☆★☆★☆★☆
############################################제출용 제작############################################
results = model.predict(test_file)       # y의 count값은
submit_file['count'] = results           # test파일에서 예측한걸 count로 나오면 submit파일에 들어가진다

print(submit_file[:10])                 # submit_file이라는 변수에 count가 들어갔고 csv로 보내준다.

submit_file.to_csv(path + "bike_submit_ver.csv", index=False)
# ★ to_csv하고 bike폴더에 생성된 파일을 보면 인덱스가 생성되어 있다. 이렇게 인덱스가 자동으로 생성되므로 index=False 옵션을 넣어줘야한다.!





'''
# 로그변환 전 후
loss :  1.484175443649292
r2스코어 :  0.24259543381440296
              datetime     count
0  2011-01-20 00:00:00  3.584190
1  2011-01-20 01:00:00  3.742803
2  2011-01-20 02:00:00  3.742803
3  2011-01-20 03:00:00  3.687594
4  2011-01-20 04:00:00  3.687594
5  2011-01-20 05:00:00  3.482172
6  2011-01-20 06:00:00  3.429094
7  2011-01-20 07:00:00  3.559641
8  2011-01-20 08:00:00  3.555471
9  2011-01-20 09:00:00  3.799100

loss :  1.4777776002883911
r2스코어 :  0.2458605468393842
RMSE :  1.2156386473737608
              datetime     count
0  2011-01-20 00:00:00  3.831663
1  2011-01-20 01:00:00  3.768396
2  2011-01-20 02:00:00  3.768396
3  2011-01-20 03:00:00  3.802023
4  2011-01-20 04:00:00  3.802023
5  2011-01-20 05:00:00  3.673296
6  2011-01-20 06:00:00  3.635793
7  2011-01-20 07:00:00  3.745045
8  2011-01-20 08:00:00  3.766439
9  2011-01-20 09:00:00  3.927194

캐글 2번째 submit
# epochs=10, 3layers&10nods
loss :  1.7997132539749146
r2스코어 :  0.08157018699570995
RMSE :  1.341533908118344
              datetime     count
0  2011-01-20 00:00:00  3.318767
1  2011-01-20 01:00:00  2.584734
2  2011-01-20 02:00:00  2.584734
3  2011-01-20 03:00:00  2.946309
4  2011-01-20 04:00:00  2.946309
5  2011-01-20 05:00:00  2.891283
6  2011-01-20 06:00:00  2.856007
7  2011-01-20 07:00:00  2.895847
8  2011-01-20 08:00:00  3.095487
9  2011-01-20 09:00:00  3.152395

[제출 결과]
Your most recent submission
Name : bike_submit_ver.csv
Submitted : 4 minutes ago
Wait time : 1 seconds
Execution time : 0 seconds
Score : 3.17983


'''