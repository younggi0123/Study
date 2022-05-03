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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, HalvingGridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)




# 1.1. 하이퍼 파라미터 ( 우리는 밑에서 randomforest를 as "rf"로 명명하였음) )
parameters = [
        {'rf__max_depth' : [6, 8, 10], 
        'rf__min_samples_leaf' : [3, 5, 7, 10]},
        {'rf__min_samples_leaf' : [3, 5, 7, 10],
         'rf__min_samples_split' : [3, 5, 10]}
]

#2. 모델
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline(  [  ("mm", MinMaxScaler()), ("rf",RandomForestClassifier())] ) # Pipeline사용법 : 1. 리스트씌우기 2.이름을 명시


# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

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
# model.score :  0.9666666666666667

# print(model)
# GridSearchCV(cv=5,
#              estimator=Pipeline(steps=[('mm', MinMaxScaler()),
#                                        ('rf', RandomForestClassifier())]),
#              param_grid=[{'rf__max_depth': [6, 8, 10]},
#                          {'rf__min_samples_leaf': [3, 5, 7, 10],
#                           'rf__min_samples_split': [2, 3, 5, 10]}],
#              verbose=1)

print("result : ", result)
# result :  0.9583333333333334

print("accuracy-score : ", acc)
# accuracy-score :  0.9666666666666667
