# ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ 
# ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ 
# ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ 
# ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ 
# ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ # ☆★ 중요한 부분 ★☆ 

# ▦ 그리드 서치 수업 !!!!!!!!!!!!

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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 파라미터별 개별로 3줄이니까
# {} 중괄호 1회
#       +
# {} 중괄호 2회
#       +
# {} 중괄호 3회

# SVC모델에 "C"라는 parameter가 있다고 한다. SVC의 kernel이라는 parameter가 있다고 한다. ...
parameters = [                                                              # 경우의 수
        {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]},    # 12 개  # 첫 번째 턴 : 1 → linear → 3, 두번째 턴 : 1 → linear → 4, ………
        {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001,0.0001]},       # 6 개
        {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],
        "gamma":[0.01, 0.001,0.0001], "degree":[3,4]}                       # 24 개
]                                                                           # 총 42 개

# 2.  모델구성
model =GridSearchCV( SVC(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1 )  # Cross Validation fit이 kfold로 된거 #verbose도 가능
                                                                            # refit을 True로 잡았을 때 가장 좋은 값을 도출해 주겠다.
                                                                            # 매개변수 안의 n_jobs는 cpu를 여러개 쓸 수 있는 기능임.
                                                                            # 괜히 150개 짜리에 n_jobs수 늘렸다간 로딩시간이 더 길어질 수 있다.
                                                                            # n-jobs = -1하면 cpu 쓰레드 다쓰겠단소리
# verbose 로: " Fitting 5 folds for each of 42 candidates, totalling 210 fits" = 42개 후보군에 대해 5fitting했고 210번 훈련함
# GridSearch할거야. gridSearch할 모데을 난 svc로 명시하고 넣은거지.
# 그모델에 맞는 parameter를 넣어야 해. 위에서 parameter는 딕셔너리 형태로 들어갔어.
#  → C를 7을 넣고 싶다? 위의 딕셔너리에서 7을 넣는거지
# model = SVC(C=1, kernel='linear', degree=3)

import os
import time

start = time.time()
# end = time.time() -start
# 머신러닝은 compile필요없으니 바로 fit
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
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

# aaa = model.score(x_test, y_test)   # .score은 evaluate개념이여!
# print(aaa)
print("Best_Score_ : ", model.best_score_)
print("Model Score : ", model.score(x_test, y_test))
# SVC모델 42번 돌아서. print하자
# 스코어 최적값(여기선 accuracy가 됨)이 0.9666666666666667이 나왔다.

y_predict = model.predict(x_test)
print("Accuracy Score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("Best Tuned ACC : ", accuracy_score(y_test, y_pred_best) )
print("=============================================================")
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')





# ######################################################################
# # print(model.cv_results_)
# # 위는 복잡하니 아래와 같이 하면 이쁘게 출력됨( 모르겠음 위에 아래 따로 출력 ㄱㄱ)
# aaa = pd.DataFrame(model.cv_results_)
# print(aaa)

# # 쓸만한 애들만 짤라서 ㄱㄱ
# bbb = aaa[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score']]                                        #평균 스코어
#         #    'split0_test_score', 'split1_test_score', 'split2_test_score', # split 0~4는 훈련 5번 시킨, cv개수만큼.    #개별 스코어
#         #    'split3_test_score', 'split4_test_score'
#         #  ]]

# print(bbb)
# # rank 1이 가장 좋은거겠죠 ?



# ╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉결론은 아래의 한줄이다.╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉
# model =GridSearchCV( SVC(), parameters, cv=kfold , verbose=1, refit=True )  # Cross Validation fit이 kfold로 된거 #verbose도 가능
# ╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉╉
# model.score를 생각하면 evaluate를 떠올린다 ☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆

# 그 외 나머지는 어떻게 돌아가는가? 로 인해 사용한 것임.
# PARAMETER는 그 모델의 gamma degree등 이런 parameter는 내가 쓰고 싶은 모델에 따라서 조절해서 써야해( xgboost 등 알아서)







# ※※※※※※※※※※※※결과값 해석※※※※※※※※※※※※※ # ※※※※※※※※※※※※결과값 해석※※※※※※※※※※※※※ #
# =============================================================
#                          결과값 확인
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# # 훈련시킨것에서 최고값 (즉,train에서 )
# Best_Score_ :  0.9916666666666668

# # Predict한 값에서 (즉, test까지 해서 )
# Model Score :  0.9916666666666667
# Accuracy Score :  0.9916666666666667
# Best Tuned ACC :  0.9916666666666667
# =============================================================