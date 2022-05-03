# GridSearchCV 적용해서 출력한 값에서
# FeatureInportace 추출 후 SelectFromModel 만들어 컬럼 축소 후
# 모델구축하여 결과 도출
# 사이킥런 load_wine 아닌 UCI _ white wine data로 ㄱㄱ




from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Normalizer, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings(action='ignore')


# 1. 데이터
path = "../_data/kaggle/wine/"
# read_csv takes a sep param, in your case, do it like so:
# data = read_csv(csv_path, sep=';')
datasets = pd.read_csv(path+"winequality-white.csv", index_col=None, header=0, sep=';')#첫째줄이 헤더고 헤더가 있음

# 정보 찍어보기
print( datasets.head() )
print( datasets.describe() )
print( datasets.info() )    #11개 컬럼 다 나온다.
                            # quality 만 int이고 y값으로 뺄 것이다. 
                            # 모든 데이터가 not-null이다. = 결측치 없다!

x = datasets.drop(['quality'], axis =1)
y = datasets['quality']
# print(x.shape, y.shape) # (4898, 11) (4898,) 11개 컬럼


# 에러발생 원인 : 판다스였어서(넘파이화해줬음)
# 밑에서 했더니 feature importance가 돌릴때마다 열 개수를 다르게 drop해줘서
# 11개가 아닐때가 생겨버려서 위로 옮겼다.
datasets = datasets.values
print(type(datasets))
print(datasets.shape)
# pandas에서 x는 드랍했었지
# x = datasets.drop(['quality'], axis =1)
# pandas에서 y는 그 컬럼만 빼왔었지
# y = datasets['quality']
# 넘파이에서는??
# 모든행, 10번쨰 열까지
x = datasets[: , :11]
y = datasets[: , 11]



# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)    #, stratify=y) regress

# Scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# 2. 모델(와인_분류)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 하이퍼 파라미터 튜닝(임의 조정)
parameters = [
    {'n_estimators' : [1000, 2000]},
    {'learning_rate' : [0.039, 0.05, 0.1, 0.2, 0.25]},
    {'max_depth' : [3, 5, 7, 9]}
]
# model = XGBClassifier(
#                     n_estimators = 2000,
#                     learning_rate = 0.039,
#                     max_depth = 7,
#                     min_child_weight = 1,
#                     subsample = 1,
#                     colsample_bytree = 1,
#                     reg_alpha = 1,  # 규제 L1
#                     reg_lambda = 0  # 규제 L2
#                     )

# 파라미터 조합으로 2개이상 엮을 것
# 모델구성
model = GridSearchCV( XGBClassifier(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1 )

############################################피쳐임포턴스 도출용 모델 생성############################################

# 3. 훈련
# model.fit( x_train, y_train )
model.fit(x_train, y_train, verbose=1,
          eval_set = [ (x_test, y_test) ],
          eval_metric='mlogloss',
          )

# 4. 평가
score = model.score( x_test, y_test )
print('model.score : ', score)

# What is the ' .best_estimator_ '   ?? when using classifier have to set best_estimator cuz its parameters are not fixed
# https://stackoverflow.com/questions/48377296/get-feature-importance-from-gridsearchcv

# 중요도 상관없이 피처 임포턴스
print(model.best_estimator_.feature_importances_)
# [0.07118042 0.11068212 0.07296681 0.08077707 0.07586595 0.08884557
#  0.06964406 0.07237581 0.07518705 0.07214403 0.21033108]

# 컬럼 순서 중요도 순으로 sort하려한다!
print(np.sort(model.best_estimator_.feature_importances_))
# [0.06964406 0.07118042 0.07214403 0.07237581 0.07296681 0.07518705
#  0.07586595 0.08077707 0.08884557 0.11068212 0.21033108]
sorted_model = np.sort(model.best_estimator_.feature_importances_)

############################################셀렉션from모델 도출용 모델 생성############################################
print("======================================================================")

acc_list = []
th_list = []

for thresh in sorted_model :
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit = True )
    select_x_train  = selection.transform(x_train)
    select_x_test  = selection.transform(x_test)
    # print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBClassifier(n_jobs = -1)

    #3 훈련
    selection_model.fit(select_x_train, y_train, eval_metric='mlogloss')

    #4 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)

    select_acc = accuracy_score(y_test, select_y_pred)

    # print("select_Score : ", score)
    print("select_Accuracy : ", select_acc)    
    # print("Thresh = %.3f, n=%d, R2 :%2f%%" %(thresh,select_x_train.shape[1],select_r2*100))
    acc_list.append(select_acc)
    th_list.append(thresh)

print(acc_list)
# [0.6795918367346939, 0.6836734693877551, 0.6704081632653062, 0.6836734693877551, 0.6704081632653062, 0.6551020408163265, 0.6479591836734694, 0.6479591836734694, 0.6010204081632653, 0.5581632653061225, 0.4887755102040816]
print(th_list)
# [0.06964406, 0.07118042, 0.07214403, 0.07237581, 0.07296681, 0.07518705, 0.07586595, 0.08077707, 0.08884557, 0.11068212, 0.21033108]

############################################셀렉션 모델 기반 컬럼축소 모델로 재생성############################################

# 중요도가 낮은 컬럼들을 체크하였으니 제거하여 재생성하고, 모델을 돌려봄

index_max_acc = acc_list.index(max(acc_list))
print(index_max_acc)    # 1
drop_list = np.where(model.best_estimator_.feature_importances_ < th_list[index_max_acc])
print(drop_list)        #(array([6], dtype=int64),)


# x = datasets.drop(['quality'], axis =1)
# y = datasets['quality']


# # 에러발생 원인 : 판다스였어서(넘파이화해줬음)
# # 여기서 했더니 feature importance가 돌릴때마다 열 개수를 다르게 drop해줘서
# # 11개가 아닐때가 생겨버리는 바람에 이 부분을 소스 윗단으로 옮겼다. ※
# datasets = datasets.values
# print(type(datasets))
# print(datasets.shape)
# # pandas에서 x는 드랍했었지
# # x = datasets.drop(['quality'], axis =1)
# # pandas에서 y는 그 컬럼만 빼왔었지
# # y = datasets['quality']
# # 넘파이에서는??
# # 모든행, 10번쨰 열까지
# x = datasets[: , :11]
# y = datasets[: , 11]

x = np.delete(x, drop_list, axis=1)

# 수정
# x = np.delete([x], drop_list, axis=1)


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
print("Score : ", score)
print("acc : ", acc)
print('==============================================')
















# 결과
######################################
# thresh

# select_Accuracy :  0.6795918367346939
# select_Accuracy :  0.6836734693877551
# select_Accuracy :  0.6704081632653062
# select_Accuracy :  0.6836734693877551
# select_Accuracy :  0.6704081632653062
# select_Accuracy :  0.6551020408163265
# select_Accuracy :  0.6479591836734694
# select_Accuracy :  0.6479591836734694
# select_Accuracy :  0.6010204081632653
# select_Accuracy :  0.5581632653061225
# select_Accuracy :  0.4887755102040816



# =================================[모델 수정 후]
# Score :  0.6602040816326531
# acc :  0.6602040816326531
# ==============================================#








# y 컬럼 축소 ㄱㄱ
# (라벨값 변경 가능한 케이스일 때(분포 줄여도 될때만 가능!)
# 3 20
# 4 163
# 5 1457
# 6 2198
# 7 880
# 8 175
# 9 5

# ex) 3과 4컬럼묶기, 8과 9컬럼 묶기 