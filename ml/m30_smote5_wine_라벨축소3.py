#    [실습1]

# 3, 4      => 0
# 5, 6, 7   => 1
# 8, 9      => 2

#       vs

# 3, 4, 5   => 0
# 6         => 1
# 7, 8, 9   => 2

# 성능 비교 !!
# // acc, f1 
# 1. SMOTE 전-후
# 2. 라벨축소 전-후 => 총 6가지 결과 도출하기




from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings(action='ignore')

# Dataset
path = "../_data/kaggle/wine/"

# Pandas DataFrame
datasets = pd.read_csv(path+"winequality-white.csv", index_col=None, header=0, sep=';')
x = datasets.drop(['quality'], axis =1)
y = datasets['quality']
# print(x.shape, y.shape) # (4898, 11) (4898,) 11개 컬럼

# PrintColumn Labels
print(pd.Series(y).value_counts())
# 6    2198
# 5    1457
# 7     880
# 8     175
# 4     163
# 3      20
# 9       5
# Name: quality, dtype: int64

# Data to Numpy 化
datasets = datasets.values
x = datasets[: , :11]
y = datasets[: , 11]

# [Column Changing Case 1]
# 3, 4      => 0
# 5, 6, 7   => 1
# 8, 9      => 2
def column_change1(y):
    for index, value in enumerate(y):
        if value == 8|9 :
            y[index] = 2
        elif value == 5|6|7 :
            y[index] = 1
        elif value == 3|4 :
            y[index] = 0
        else:
            y[index] = 0
    return y

# [Column Changing Case 2]
# 3, 4, 5   => 0
# 6         => 1
# 7, 8, 9   => 2
def column_change2(y):
    for index, value in enumerate(y):
        if value == 7|8|9 :
            y[index] = 2
        elif value == 6 :
            y[index] = 1
        elif value == 3|4|5 :
            y[index] = 0
        else:
            y[index] = 0
    return y
# Choose Column_set yourself
# y = column_change1(y) # no.1
y = column_change2(y) # no.2
print(pd.Series(y).value_counts())    

# no.1
# 0.0    4013
# 1.0     880
# 2.0       5

# no.2
# 0.0    2700
# 1.0    2198




# SPLIT
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)    #, stratify=y) regress

# SCALER
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# MODEL (Wine_ Classification)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# HyperParameter tuning
parameters = [
    {'n_estimators' : [1000, 2000]},
    {'learning_rate' : [0.039, 0.05, 0.1, 0.2, 0.25]},
    {'max_depth' : [3, 5, 7, 9]}
]

model = GridSearchCV( XGBClassifier(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1 )

# FIT
model.fit(x_train, y_train, verbose=1,
          eval_set = [ (x_test, y_test) ],
          eval_metric='mlogloss',
          )

# EVALUATE
score = model.score( x_test, y_test )
print('model.score : ', score)

# FEATURE IMPORTANCE
# Print Unsorted F.I.
print(model.best_estimator_.feature_importances_)

# Print Sorted F.I.
print(np.sort(model.best_estimator_.feature_importances_))

# Make Sorted_Model
sorted_model = np.sort(model.best_estimator_.feature_importances_)

# Set Thresh Modeling Part
print("======================================================================")

acc_list = []
th_list = []
for thresh in sorted_model :
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit = True )
    select_x_train  = selection.transform(x_train)
    select_x_test  = selection.transform(x_test)
    
    selection_model = XGBClassifier(n_jobs = -1)

    # 훈련
    selection_model.fit(select_x_train, y_train, eval_metric='mlogloss')

    # 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)
    select_acc = accuracy_score(y_test, select_y_pred)

    print("select_Accuracy : ", select_acc)    
    # print("Thresh = %.3f, n=%d, R2 :%2f%%" %(thresh,select_x_train.shape[1],select_r2*100))
    acc_list.append(select_acc)
    th_list.append(thresh)
    
# Select Acc
print(acc_list)

# Thresh
print(th_list)

# Drop Columns
index_max_acc = acc_list.index(max(acc_list))
print(index_max_acc)    # 1
drop_list = np.where(model.best_estimator_.feature_importances_ < th_list[index_max_acc])
print(drop_list)        #(array([6], dtype=int64),)
x = np.delete(x, drop_list, axis=1)



#####################################################################################
#                                  ※ 기본 모델 ※
#####################################################################################

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
model.fit(x_train, y_train, verbose=1,
          eval_set = [ (x_test, y_test) ],
          eval_metric='mlogloss',
          )

#4 평가 예측
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print('================================= 모델 수정 후')
acc = accuracy_score(y_test, y_pred)
print("Score : ", round(score,4))
print("acc : ", round(acc,4))
print('F1_score : ',round(f1_score(y_test, y_pred, average='macro'),4 ))
print('==============================================')