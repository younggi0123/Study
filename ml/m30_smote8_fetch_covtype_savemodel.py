import warnings

from lightgbm import early_stopping
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

# 시간체크
import os
import time
start = time.time()

# 증폭된 데이터셋 넘파이 로드
path = '../_save/'
train_datasets = np.load(path + 'm30_savedata_fetch_covtype_smoted_trains.npz')
test_datasets = np.load(path + 'm30_savedata_fetch_covtype_tests.npz')

for i in train_datasets:
    print(i)            # x, y로 출력해봄
x_train = train_datasets['x']
y_train = train_datasets['y']

for i in test_datasets:
    print(i)            # x, y로 출력해봄
x_test = test_datasets['x']
y_test = test_datasets['y']

print(x_train.shape, y_train.shape) # (1586480, 54) (1586480,)
# 증폭된 y 컬럼 라벨별 행개수 파악
print(np.unique(y_train, return_counts=True))
# array([1,      2,       3,       4,      5,     6,      7    ])
#        ▼       ▼        ▼        ▼       ▼      ▼       ▼
# array([226640, 226640,  226640,  226640, 226640,226640, 226640], dtype=int64) )

# Scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# HyperParameterTuning
parameters = [{
    'booster' : ['gbtree'] ,
    'n_estimators' : [5000] ,
    'learning_rate' : [0.05, 0.08, 0.12, 0.2] ,
    'max_depth' : [3, 5, 7] ,
     # GPU연산하겠다!
    'tree_method' : ['gpu_hist'] ,
    'predictor' : ['gpu_predictor'] ,
    'gpu_id' : [0]
    } ]

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
model = RandomizedSearchCV( XGBClassifier(), parameters, cv=kfold , verbose=1, refit=True)#, n_jobs= -1 )

#####################################################################################################################
############################################피쳐임포턴스 도출용 모델 생성############################################
#####################################################################################################################

# 3. 훈련
model.fit(x_train, y_train, verbose=1,
eval_set = [ (x_test, y_test) ],
eval_metric='mlogloss',
early_stopping=100)

# 4. 평가
score = model.score( x_test, y_test )
print('model.score : ', score)
# model.score :  0.7723450658060612

# 중요도 상관없이 피처 임포턴스
print(model.best_estimator_.feature_importances_)
# [0.09728908 0.00588848 0.00272104 0.01802587 0.00544451 0.00612319
#  0.01098906 0.00602844 0.00849645 0.00765417 0.01105719 0.00235593
#  0.01295023 0.01510703 0.00183675 0.01675552 0.12844609 0.02851299
#  0.0094664  0.01713877 0.         0.         0.00044128 0.19472149
#  0.00162101 0.12528792 0.06861984 0.         0.         0.00049211
#  0.         0.         0.         0.00213157 0.         0.00434058
#  0.00729748 0.00290119 0.         0.         0.00082732 0.00062371
#  0.00787322 0.08493694 0.00339446 0.00860517 0.00527739 0.00035736
#  0.00517999 0.         0.         0.01817542 0.04460735 0.        ]

# 컬럼 순서 중요도 순으로 sort하려한다!
print(np.sort(model.best_estimator_.feature_importances_))
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.00035736 0.00044128 0.00049211 0.00062371 0.00082732
#  0.00162101 0.00183675 0.00213157 0.00235593 0.00272104 0.00290119
#  0.00339446 0.00434058 0.00517999 0.00527739 0.00544451 0.00588848
#  0.00602844 0.00612319 0.00729748 0.00765417 0.00787322 0.00849645
#  0.00860517 0.0094664  0.01098906 0.01105719 0.01295023 0.01510703
#  0.01675552 0.01713877 0.01802587 0.01817542 0.02851299 0.04460735
#  0.06861984 0.08493694 0.09728908 0.12528792 0.12844609 0.19472149]

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
    
    selection_model = XGBClassifier(
                    # GPU연산하겠다!
                    tree_method = 'gpu_hist',
                    predictor = 'gpu_predictor',
                    gpu_id = 0)

    #3 훈련
    selection_model.fit(select_x_train, y_train, eval_metric='mlogloss')

    #4 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)

    # select_acc = accuracy_score(y_test, select_y_pred)
    select_f1 = f1_score(y_test, select_y_pred, average='macro')

    # print("select_Score : ", score)
    print("select_F1 : ", select_f1)
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
x = np.delete(x_train, drop_list, axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_train, 
         train_size = 0.8, shuffle = True, random_state = 66 , stratify= y_train
         )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 모델
# model = XGBClassifier(
#                     # GPU연산하겠다!
#                     tree_method = 'gpu_hist',
#                     predictor = 'gpu_predictor',
#                     gpu_id = 0)
model = RandomizedSearchCV( XGBClassifier(), parameters, cv=kfold , verbose=1, refit=True) #, n_jobs= -1 )

#3 훈련
model.fit(x_train, y_train, verbose=1,
          eval_set = [ (x_test, y_test) ],
          eval_metric='mlogloss',
          )

#4 평가 예측
score = model.score(x_test, y_test)
print("model.score : ", round(score,4) )
y_predict = model.predict(x_test)
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ) )
print('f1_score : ',
      round(f1_score(y_test, y_predict, average='macro'), 4) )
# print('f1_score : ',f1_score(y_test, y_predict, average='micro'))

import joblib
joblib.dump(model, path + "m30_savemodel_fetch_covtype.dat")

print("걸린시간 : ", time.time()-start)

try:
    path = '../_save/'
    # np.savez(path + 'm30_savemodel_fetch_covtype.npz') # 각각 이름 붙여줘야 불러오기 가능
except:
    print("이건 안되네..")