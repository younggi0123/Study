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
train_datasets = np.load(path + 'm30_savedata_fetch_covtype_smoted_trains.npz')
# train_datasets = np.load(path + 'm30_savedata_fetch_covtype.npz')
test_datasets = np.load(path + 'm30_savedata_fetch_covtype_tests.npz')
###############################################################
x_train = train_datasets['x']
y_train = train_datasets['y']
x_test = test_datasets['x']
y_test = test_datasets['y']
###############################################################

# 모델 로드
import joblib
path = '../_save/'
# model = joblib.load(path + "m30_savemodel_fetch_covtype.dat")
model = joblib.load(path + "m30_savemodel_fetch_covtype2.dat")
# model = joblib.load(path + "m30_savemodel_fetch_covtype3.dat")
# model = joblib.load(path + "fetch_save2.dat")


score = model.score(x_test, y_test)
print("model.score : ", round(score,4) )
y_predict = model.predict(x_test)
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ) )
print('f1_score : ',
      round(f1_score(y_test, y_predict, average='macro'), 4) )
# print('f1_score : ',f1_score(y_test, y_predict, average='micro'))