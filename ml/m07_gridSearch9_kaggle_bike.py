# 실습

# 모델 사용 : RandomForestClassifier

from tkinter import Grid
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pandas import read_csv

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = read_csv(path+'train.csv')  
# test_file = read_csv(path+'test.csv')
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
# test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 하이퍼 파라미터 튜닝
parameters1 = [
    {'n_estimators' : [100, 200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3,5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

parameters2 = [
    {'n_estimators' : [100, 200],
    'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3,5, 7, 10],
    'min_samples_split' : [2, 3, 5, 10]},
]
# 파라미터 조합으로 2개이상 엮을 것
# 2.  모델구성
model =GridSearchCV( RandomForestClassifier(), parameters2, cv=kfold , verbose=1, refit=True, n_jobs= -1 )

import os, time
start = time.time()


# 3. fit
model.fit(x_train, y_train)

# 4. 평가, 예측
x_test= x_train     # 과적합 상황 보여주기
y_test = y_train    # train데이터로 best_estimator_로 예측 뒤 점수를 내면
                    # best_score_ 나온다.

print("=============================================================")
print("                       [결과값 확인]                         ")
            # 최적값 눈으로 확인 !
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
print("걸린시간 : ", round(end, 2), '초')

# Fitting 5 folds for each of 24 candidates, totalling 120 fits
# =============================================================
#                        [결과값 확인]
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 6, 'n_estimators': 200}
# Best_Score_ :  0.01837338294169589
# Model Score :  0.10679834634818558
# Accuracy Score :  0.10679834634818558
# Best Tuned ACC :  0.10679834634818558
# =============================================================
# 걸린시간 :  86.45 초