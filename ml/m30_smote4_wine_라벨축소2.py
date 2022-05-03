# GridSearchCV 적용해서 출력한 값에서
# FeatureInportace 추출 후 SelectFromModel 만들어 컬럼 축소 후
# 모델구축하여 결과 도출
# 사이킥런 load_wine 아닌 UCI _ white wine data로 ㄱㄱ


############### 컬럼축소 실습용 #################################

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


from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Normalizer, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score
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
print(x.shape, y.shape) # (4898, 11) (4898,) 11개 컬럼


# y(quality) 컬럼 출력
count_data = datasets.groupby("quality")['quality'].count()  
print(count_data) # 이방법 혹은
print(np.unique(y, return_counts=True)) # 이 방법
from pprint import pprint   # 이건 그냥 이쁘게 출력하는법
# pprint(np.unique(y, return_counts=True))

# 에러발생 원인 : 판다스였어서(넘파이화해줬음)
# 밑에서 했더니 feature importance가 돌릴때마다 열 개수를 다르게 drop해줘서
# 11개가 아닐때가 생겨버려서 위로 옮겼다.
datasets = datasets.values
# print(type(datasets))
# print(datasets.shape)
# pandas에서 x는 드랍했었지
# x = datasets.drop(['quality'], axis =1)
# pandas에서 y는 그 컬럼만 빼왔었지
# y = datasets['quality']
# 넘파이에서는??
# 모든행, 10번쨰 열까지
x = datasets[: , :11]
y = datasets[: , 11]





############################################################################################
##################################### y값 축소부분##########################################
# if value == 9 :             를
        # y[index] = 8        로 바꿨더니 성능이 0.03이나 향상되었다!
# elif value == 8 :
        # y[index] = 8        로 바꿨더니 위보다 성능이 0.05이나 향상되었다!
# elif value == 3 :
        # y[index] = 4        로 바꿨더니 위보다 성능이 0.1이나 향상되었다!
############################################################################################

# # 1.2. 선생님 ver.
for index, value in enumerate(y):
    if value == 9 :
        y[index] = 9
    elif value == 8 :
        y[index] = 8
    elif value == 7 :
        y[index] = 7
    elif value == 6 :
        y[index] = 6
    elif value == 5 :
        y[index] = 5
    elif value == 4 :
        y[index] = 4
    elif value == 3 :
        y[index] = 3
    else:
        y[index] = 0
print(pd.Series(y).value_counts())    
# 1.0    4893
# 8.0       5








# 2. 나 ver.1.
#     └ y값 자체를 변경하고자 함
# for i in y:
#     if i<=4 :
#         y[:, i] == 0
#     elif i<=7:
#         y[:, i] == 1
#         #only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
#     else:
#         y[:, i] == 2
# print(np.unique(y, return_counts=True))

# # 2.1. 나 ver.2.
# def func_change(y):
#         yg_list = []
#         if y<=4:
#             yg_list += [0]
#         elif y<=7:
#             yg_list += [1]
#         else:
#             yg_list += [2]
#         y= np.array(newlist)
  
# for i in y:
#     result = list(map( ))
# y= np.array(newlist)



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



# 컬럼 순서 중요도 순으로 sort하려한다!
print(np.sort(model.best_estimator_.feature_importances_))



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
x = np.delete(x, drop_list, axis=1)



#####################################################################################
#                                     ※ 기본 ※
#####################################################################################




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
print("Score : ", round(score,4))
print("acc : ", round(acc,4))
print('F1_score : ',round(f1_score(y_test, y_pred, average='macro'),4 ))
print('==============================================')

#####################################################################################
#####################################################################################





# 결과
######################################
# [thresh]
# select_Accuracy :  0.9469387755102041
# select_Accuracy :  0.9479591836734694
# select_Accuracy :  0.9459183673469388
# select_Accuracy :  0.9428571428571428
# select_Accuracy :  0.9387755102040817
# select_Accuracy :  0.939795918367347
# select_Accuracy :  0.9418367346938775
# select_Accuracy :  0.9387755102040817
# select_Accuracy :  0.9244897959183673
# select_Accuracy :  0.9193877551020408
# select_Accuracy :  0.9255102040816326
# ================================= 모델 수정 후
# Score :  0.9398
# acc :  0.9398
# F1_score :  0.6247
# ==============================================












#####################################################################################
#                                  ※ SMOTE적용 ※
#####################################################################################



# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=66) # , k_neighbors=2)
# x_train, y_train = smote.fit_resample(x_train, y_train) # fit resample에 xtrain ytrain 넣은 것을 반환해줘야
# # print( pd.Series(y_train).value_counts() )

# #2 모델
# model = XGBClassifier(n_jobs = -1)

# #3 훈련
# model.fit(x_train, y_train, verbose=1,
#           eval_set = [ (x_test, y_test) ],
#           eval_metric='mlogloss',
#           )

# #4 평가 예측
# score = model.score(x_test, y_test)
# y_pred = model.predict(x_test)
# print('================================= 모델 수정 후')
# acc = accuracy_score(y_test, y_pred)
# print("Score : ", score)
# print("Acc : ", acc)
# print('F1_score : ',f1_score(y_test, y_pred, average='macro'))
# print('==============================================')



# # 결과
# ######################################
# # [thresh]
# # select_Accuracy :  0.9469387755102041
# # select_Accuracy :  0.9479591836734694
# # select_Accuracy :  0.9459183673469388
# # select_Accuracy :  0.9428571428571428
# # select_Accuracy :  0.9387755102040817
# # select_Accuracy :  0.939795918367347
# # select_Accuracy :  0.9418367346938775
# # select_Accuracy :  0.9387755102040817
# # select_Accuracy :  0.9244897959183673
# # select_Accuracy :  0.9193877551020408
# # select_Accuracy :  0.9255102040816326
# # ================================= 모델 수정 후
# # Score :  0.9448979591836735
# # Acc :  0.9448979591836735
# # F1_score :  0.717050547622927
# # ==============================================

######################################################################################
######################################################################################