
import numpy as np
import pandas as pd
from random import random
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

datasets = load_wine()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
        {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3,5, 7, 10], 'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]},
        {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3,5, 7, 10], 'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]},
        {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3,5, 7, 10], 'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}
]

# 2.  모델구성
model = RandomizedSearchCV( RandomForestClassifier(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1, random_state=66, n_iter=20 )
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



# =============================================================
#                        [결과값 확인]
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=5,
#                        n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_depth': 6}
# Best_Score_ :  0.9714285714285715
# Model Score :  1.0
# Accuracy Score :  1.0
# Best Tuned ACC :  1.0
# =============================================================
# 걸린시간 :  6.086 초