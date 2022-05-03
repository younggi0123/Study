#####################################################################################################################
# 패치코브 커피 경쟁: fetch_covtype
# F1-score-macro기준 열변경가능 행변경x 이외 아무방법 다가능 !
# 30-8소스 : 데이터 증폭, 저장(피클 or 넘파이)
# 30-9소스 : 데이터 로드, 코드완성
# 참고로,
# 증폭해서 판다스든 피클이든 저장하고 그 파일로 돌려야지
# 할때마다 증폭하면 시간낭비 핵오바이다.
#####################################################################################################################
import warnings
warnings.filterwarnings(action='ignore')

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel

# 증폭된 데이터셋 넘파이 로드
path = '../_save/'
datasets = np.load(path + 'm30_pickle_savedata_fetch_covtype.npz')

for i in datasets:
    print(i)            # x, y로 출력해봄
x = datasets['x']
y = datasets['y']
print(x.shape, y.shape) # (1586480, 54) (1586480,)
# 증폭된 y 컬럼 라벨별 행 파악
print(np.unique(y))     # [1 2 3 4 5 6 7]
# 증폭된 y 컬럼 라벨별 행개수 파악
print(np.unique(y, return_counts=True))
# array([1,      2,       3,       4,      5,     6,      7    ])
#        ▼       ▼        ▼        ▼       ▼      ▼       ▼
# array([226640, 226640,  226640,  226640, 226640,226640, 226640], dtype=int64) )

# Split
x_train, x_test, y_train, y_test \
    = train_test_split(x, y, train_size=0.8,
    shuffle=True, random_state=66, stratify=y )

# Scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# HyperParameterTuning
parameters = [
    {'n_estimators' : [1]},
    # {'learning_rate' : [0.05, 0.1, 0.25]},
    # {'max_depth' : [3, 5, 7]}
]
# XGBClassifier_Parameters 참고용
# n_estimators = 2000,
# learning_rate = 0.039,
# max_depth = 7,
# min_child_weight = 1,
# subsample = 1,
# colsample_bytree = 1,
# reg_alpha = 1,  # 규제 L1
# reg_lambda = 0  # 규제 L2

# 모델 구성
model = RandomizedSearchCV( XGBClassifier(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1 )

#####################################################################################################################
############################################피쳐임포턴스 도출용 모델 생성############################################
#####################################################################################################################

# 3. 훈련
weight_path = "../_save/feature_importance_fetch_covtype_weight.h5"
if os.path.exists(weight_path):
    model.best_estimator_.load_weights(weight_path)
#model = load_model(path)  


else:
    model.fit(x_train, y_train, verbose=1,
        eval_set = [ (x_test, y_test) ],
        eval_metric='mlogloss',
        )
    model.best_estimator_.save("../_save/feature_importance_fetch_covtype_weight.h5")


# 4. 평가
score = model.score( x_test, y_test )
print('model.score : ', score)
# model.score :  0.7723450658060612

# 중요도 상관없이 피처 임포턴스
print(model.best_estimator_.feature_importances_)
# [0.10763563 0.00540047 0.00364297 0.01687661 0.00462584 0.0074327
#  0.01152723 0.00607856 0.00676409 0.0077324  0.01479799 0.00447797
#  0.01190201 0.01264811 0.00222938 0.02253913 0.1441799  0.03297993
#  0.01309747 0.01683837 0.         0.         0.00043277 0.16929005
#  0.00168286 0.06110677 0.05961402 0.00034525 0.         0.00048777
#  0.00093912 0.         0.00271541 0.00309518 0.         0.00752345
#  0.00835878 0.00323493 0.         0.         0.00086524 0.00091986
#  0.00901774 0.12064967 0.00329214 0.00888092 0.00547165 0.00048529
#  0.         0.         0.         0.02569541 0.04092114 0.01156793]

# 컬럼 순서 중요도 순으로 sort하려한다!
print(np.sort(model.best_estimator_.feature_importances_))
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.00034525 0.00043277
#  0.00048529 0.00048777 0.00086524 0.00091986 0.00093912 0.00168286
#  0.00222938 0.00271541 0.00309518 0.00323493 0.00329214 0.00364297
#  0.00447797 0.00462584 0.00540047 0.00547165 0.00607856 0.00676409
#  0.0074327  0.00752345 0.0077324  0.00835878 0.00888092 0.00901774
#  0.01152723 0.01156793 0.01190201 0.01264811 0.01309747 0.01479799
#  0.01683837 0.01687661 0.02253913 0.02569541 0.03297993 0.04092114
#  0.05961402 0.06110677 0.10763563 0.12064967 0.1441799  0.16929005]

sorted_model = np.sort(model.best_estimator_.feature_importances_)

############################################셀렉션from모델 도출용 모델 생성############################################
print("======================================================================")

f1_list = []
th_list = []
for thresh in sorted_model :
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit = True )
    select_x_train  = selection.transform(x_train)
    select_x_test  = selection.transform(x_test)
    # print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBClassifier(n_jobs = -1)

    #3 훈련
    weight_path = "../_save/featured_fetch_covtype_weight.h5"
    if os.path.exists(weight_path):
        model.best_estimator_.load_weights(weight_path)
    #model = load_model(path)  
    else:
        selection_model.fit(select_x_train, y_train, eval_metric='mlogloss')
        model.best_estimator_.save("../_save/featured_fetch_covtype_weight.h5")

    #4 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)

    # select_acc = accuracy_score(y_test, select_y_pred)
    select_f1 = f1_score(y_test, select_y_pred, average='macro')

    # print("select_Score : ", score)
    print("select_Accuracy : ", select_f1)
    # print("Thresh = %.3f, n=%d, R2 :%2f%%" %(thresh,select_x_train.shape[1],select_r2*100))
    f1_list.append(select_f1)
    th_list.append(thresh)

print(f1_list)
print(th_list)


############################################셀렉션 모델 기반 컬럼축소 모델로 재생성############################################

# 중요도가 낮은 컬럼들을 체크하였으니 제거하여 재생성하고, 모델을 돌려봄
# Drop columns
index_max_f1 = f1_list.index(max(f1_list))
print(index_max_f1)    # 9
drop_list = np.where(model.best_estimator_.feature_importances_ < th_list[index_max_f1])
print(drop_list)        #(array([14, 18, 20, 21, 28, 32, 38, 41, 49], dtype=int64),)
x = np.delete(x, drop_list, axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66 , stratify= y
         )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 모델
model = XGBClassifier(n_jobs = -1)

#3 훈련
weight_path = "../_save/selected_fetch_covtype_weight.h5"
if os.path.exists(weight_path):
    model.load_weights(weight_path)
#model = load_model(path)  
else:
    model.fit(x_train, y_train, verbose=1,
          eval_set = [ (x_test, y_test) ],
          eval_metric='mlogloss',
          )
    model.save("../_save/selected_fetch_covtype_weight.h5")

#4 평가 예측

score = model.score(x_test, y_test)
print("model.score : ", round(score,4) )
y_predict = model.predict(x_test)
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ) )
print('f1_score : ',
      round(f1_score(y_test, y_predict, average='macro'), 4) )
# print('f1_score : ',f1_score(y_test, y_predict, average='micro'))