# 파이프라인 + 그리드서치  ( ※끝판왕※ )

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

# 1. 데이터
datasets = fetch_covtype()
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
pipe = Pipeline(  [  ("mm", StandardScaler()), ("rf",RandomForestClassifier())] ) # Pipeline사용법 : 1. 리스트씌우기 2.이름을 명시


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
