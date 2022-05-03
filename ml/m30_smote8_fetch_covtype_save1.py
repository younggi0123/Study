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

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import os,time

#1. 데이터
datasets = fetch_covtype() 
x = datasets.data
y = datasets.target
print(x.shape,y.shape)  # (581012, 54) (581012,)
# y 컬럼 라벨별 행 파악
print(np.unique(y))     # [1 2 3 4 5 6 7]
# y라벨별 행개수 판단
print(np.unique(y, return_counts=True))
# array([1,      2,       3,       4,      5,     6,      7    ])
#        ▼       ▼        ▼        ▼       ▼      ▼       ▼
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64) )

# 위를 바탕으로, 211,840개를 기준삼아 데이터를 증폭(SMOTE)할 것이다.

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8, stratify=y)

start = time.time()
# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# 증폭된 y 컬럼 라벨별 행개수 파악
print(np.unique(y, return_counts=True))
# array([1,      2,       3,       4,      5,     6,      7    ])
#        ▼       ▼        ▼        ▼       ▼      ▼       ▼
# array([226640, 226640,  226640,  226640, 226640,226640, 226640], dtype=int64) )

# 증폭된 데이터셋 넘파이 저장+시간체크
# https://seong6496.tistory.com/142

path = '../_save/'
np.savez(path + 'm30_savedata_fetch_covtype_tests.npz', x=x_test, y=y_test) # 각각 이름 붙여줘야 불러오기 가능
np.savez(path + 'm30_savedata_fetch_covtype_smoted_trains.npz', x=x_train, y=y_train) # 각각 이름 붙여줘야 불러오기 가능
print("걸린시간 : ", time.time()-start)