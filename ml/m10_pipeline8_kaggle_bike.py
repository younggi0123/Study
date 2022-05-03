import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = read_csv(path+'train.csv')  
# test_file = read_csv(path+'test.csv')
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
# test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)

# 2. 모델구성
from sklearn.ensemble import RandomForestClassifier

# 파이프라인 사용 방법
from sklearn.pipeline import make_pipeline, Pipeline


#2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())

#3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("result : ", result)
# result :  0.844281120808452

print("accuracy-score : ", acc)
# accuracy-score :  0.005050505050505051