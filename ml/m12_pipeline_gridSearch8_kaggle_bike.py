# 파이프라인 + 그리드서치  ( ※끝판왕※ )

# 파이프 라인과 별개의 xgboost관련 설명 :
# xgboost의 핵심 라이브러리는 C/ C++로 작성되어 있기 때문에 초기에는 사이킷런과 호환되지 않았습니다.
# 이 말은 사이킷런의 fit(), predict() 메서드들과 GridSearchCV 등의 유틸리티를 사용할 수 없다는 것입니다.
# 그래서 xgboost를 파이썬에서 구현하는 방법으로 별도의 API를 사용하였습니다.
# 하지만 파이썬 기반의 머신러닝 이용자들이 점차 늘어나면서 xgboost 개발 그룹에서 사이킷런과 호환이 가능하도록 래퍼 클래스(wrapper class)를 제공하게 되었고, 결국 xgboost를 사용하는데에 두 가지 방법이 생기게 되었습니다
# 아래 블로그를 꼭 읽어보길.
# https://hwi-doc.tistory.com/entry/%EC%9D%B4%ED%95%B4%ED%95%98%EA%B3%A0-%EC%82%AC%EC%9A%A9%ED%95%98%EC%9E%90-XGBoost

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
# 방법 1.
# import xgboost as xgb
# 방법 2. ▼하단▼
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from pandas import read_csv

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = read_csv(path+'train.csv')  
# test_file = read_csv(path+'test.csv')
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
# test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, HalvingGridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)




# 1.1. 하이퍼 파라미터 ( 우리는 밑에서 randomforest를 as "rf"로 명명하였음) )
# parameters = [
#         {'rf__max_depth' : [6, 8, 10], 
#         'rf__min_samples_leaf' : [3, 5, 7, 10]},
#         {'rf__min_samples_leaf' : [3, 5, 7, 10],
#          'rf__min_samples_split' : [3, 5, 10]}
# ]

parameters = [
        {'xgbclassifier__n_estimators' : [100, 200], 'xgbclassifier__max_depth' : [6, 8, 10, 12], 'xgbclassifier__min_child_weight' : [1, 2, 3, 4], 'xgbclassifier__gamma' : [0.001,0.0001, 0.0001]},
        {'xgbclassifier__n_estimators' : [100, 200], 'xgbclassifier__max_depth' : [6, 8, 10, 12], 'xgbclassifier__min_child_weight' : [1, 2, 3, 4], 'xgbclassifier__gamma' : [0.001,0.0001, 0.0001]},
        {'xgbclassifier__n_estimators' : [100, 200], 'xgbclassifier__max_depth' : [6, 8, 10, 12], 'xgbclassifier__min_child_weight' : [1, 2, 3, 4], 'xgbclassifier__gamma' : [0.001,0.0001, 0.0001]}
]

#2. 모델
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
# pipe = Pipeline(  [  ("mm", MinMaxScaler()), ("rf",RandomForestClassifier())  ] ) # Pipeline사용법 : 1. 리스트씌우기 2.이름을 명시
# pipe = Pipeline(  [  ("mm", MinMaxScaler()), ("xgb", XGBClassifier())   ] ) # Pipeline사용법 : 1. 리스트씌우기 2.이름을 명시
pipe = make_pipeline(MinMaxScaler(), XGBClassifier())

# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

#3. 훈련
import os, time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
result = model.score(x_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("걸린시간 : ", end- start)
print("model.score : ", result)
print("accuracy_score : ", acc)


# 얼토달토 없는 결과의 이유
# https://kin.naver.com/qna/detail.nhn?d1id=1&dirId=10402&docId=379921126&qb=dGVzdDI=&enc=utf8&section=kin.qna&rank=628&search_sort=0&spq=0

# 걸린시간 :  391.49000120162964
# model.score :  0.09416628387689481
# accuracy_score :  0.015610651974288337

# 걸린시간 :  78.63117241859436
# model.score :  0.26056499770326136
# accuracy_score :  0.012396694214876033

# # XGBClassifier
# 걸린시간 :  3845.8474955558777
# model.score :  0.3987138263665595
# accuracy_score :  0.008723599632690543