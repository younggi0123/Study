# [랜덤 서치] !!!!!!!!!!!!
#  => parameter 숫자를 줄여버림
# 랜덤은 n_iter의 개수만 쓰겠다 parameter에서 임의로 뽑는것.



# import랑 model 바꾸고 parameter 경우의 수를 늘려줬음 => 성능 유사, 속도 up
# n_iter조절로 파라미터 검색 횟수 조절 => 20* 5 = 100

from random import random
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

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score

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
model = RandomizedSearchCV( SVC(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1, random_state=66, n_iter=20 )
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


# print(model.cv_results_)
# aaa = pd.DataFrame(model.cv_results_)
# print(aaa)

# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# =============================================================
#                        [결과값 확인]
# 최적의 매개변수 :  SVC(C=1, degree=4, gamma=0.001, kernel='linear')
# 최적의 파라미터 :  {'kernel': 'linear', 'gamma': 0.001, 'degree': 4, 'C': 1}
# Best_Score_ :  0.9916666666666668
# Model Score :  0.9916666666666667
# Accuracy Score :  0.9916666666666667
# Best Tuned ACC :  0.9916666666666667
# =============================================================
# 걸린시간 :  2.67 초



# verbose=1 로 찍어봤음
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
#  → 50개중 10개의 후보군으로 5fold하겠단 말 !
