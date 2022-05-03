# [Halving 서치 - 데이터 일부]

# halving은 100%쓰지않고 최소량만 쓰겠다
# 예를들면 상위 30%로 빼서 데이터를 100%돌려 젤 좋은 값을 뽑겠다.
#parameter = 40
# CV =5	  총200
# 데이터의 일부만 돌려서 상위 랭커만 뽑는다 (데이터의 일부를 돌리고 그중에서 상위값만 뽑아서)

from random import random
from tkinter import Grid
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

datasets = load_iris()
x = datasets.data
y = datasets.target
from sklearn.experimental import enable_halving_search_cv       #얘가 halving보다 더 위에있어야
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold, cross_val_score, HalvingGridSearchCV
# HalvingGridSearch는 완성되지 않은 버전이므로 아래와 같은 사이킥런의 실험적 실행을 추가해줘야한다


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
        {"C":[1, 10, 100, 1000], "kernel":["linear"], "gamma":[0.001,0.0001, 0.0001], "degree":[3,4,5,6]},
        {"C":[1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001,0.0001, 0.0001], "degree":[3,4,5,6]},
        {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],
        "gamma":[0.01, 0.001,0.0001, 0.0001], "degree":[3,4,5,6]}
]

# 2.  모델구성
# model = GridSearchCV( SVC(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1 )
# model = RandomizedSearchCV( SVC(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1, random_state=66, n_iter=20 )
model = HalvingGridSearchCV( SVC(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1, random_state=66)

# n_iter조절로 => 20* 5 = 100

import os, time
start = time.time()

# 3. fit
model.fit(x_train, y_train)

# 4. 평가, 예측
x_test= x_train
y_test = y_train

print("=============================================================")
print("                       [결과값 확인]                         ")
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print("Best_Score_ : ", model.best_score_)
print("Model Score : ", model.score(x_test, y_test))


y_predict = model.predict(x_test)
print("Accuracy Score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("Best Tuned ACC : ", accuracy_score(y_test, y_pred_best) )
print("=============================================================")

end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

