# GridSearchCV 적용해서 출력한 값에서
# FeatureInportace 추출 후 SelectFromModel 만들어 컬럼 축소 후
# 모델구축하여 결과 도출



from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Normalizer, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터

datasets = fetch_covtype() 

x = datasets.data
y = datasets.target
# print(datasets.feature_names)
# print(x.shape,y.shape)    #(581012, 54) (581012,)


# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)    #, stratify=y) regress

# Scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# 2. 모델(분류)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 하이퍼 파라미터 튜닝(임의 조정)
parameters = [
    {'n_estimators' : [2]},
    # {'learning_rate' : [0.039, 0.05, 0.1, 0.2, 0.25]},
    # {'max_depth' : [3, 5, 7, 9]}
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
model = RandomizedSearchCV( XGBClassifier(), parameters, cv=kfold , verbose=1, refit=True, n_jobs= -1 )

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
# model.score :  0.7974148688071736

# What is the ' .best_estimator_ '   ?? when using classifier have to set best_estimator cuz its parameters are not fixed
# https://stackoverflow.com/questions/48377296/get-feature-importance-from-gridsearchcv

# 중요도 상관없이 피처 임포턴스
print(model.best_estimator_.feature_importances_)
# [0.12446197 0.00740106 0.00389339 0.01572655 0.00648663 0.01361222
#  0.01007581 0.01512577 0.00566702 0.01280762 0.05410932 0.0253611
#  0.02844274 0.01206515 0.00201346 0.05692217 0.02964839 0.03785131
#  0.00282062 0.00385705 0.         0.         0.00557887 0.01276154
#  0.00878943 0.04626995 0.00620605 0.0044468  0.         0.00542551
#  0.00947379 0.00569033 0.00262008 0.01650353 0.01765533 0.0591464
#  0.0397611  0.01448517 0.         0.00977672 0.01820237 0.0004293
#  0.03487859 0.020028   0.02298802 0.050168   0.01748972 0.00447131
#  0.01727635 0.         0.01356674 0.02767258 0.02716358 0.01272543]

# 컬럼 순서 중요도 순으로 sort하려한다!
print(np.sort(model.best_estimator_.feature_importances_))
# [0.         0.         0.         0.         0.         0.0004293
#  0.00201346 0.00262008 0.00282062 0.00385705 0.00389339 0.0044468
#  0.00447131 0.00542551 0.00557887 0.00566702 0.00569033 0.00620605
#  0.00648663 0.00740106 0.00878943 0.00947379 0.00977672 0.01007581
#  0.01206515 0.01272543 0.01276154 0.01280762 0.01356674 0.01361222
#  0.01448517 0.01512577 0.01572655 0.01650353 0.01727635 0.01748972
#  0.01765533 0.01820237 0.020028   0.02298802 0.0253611  0.02716358
#  0.02767258 0.02844274 0.02964839 0.03487859 0.03785131 0.0397611
#  0.04626995 0.050168   0.05410932 0.05692217 0.0591464  0.12446197]


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
print(th_list)




############################################셀렉션 모델 기반 컬럼축소 모델로 재생성############################################

# 중요도가 낮은 컬럼들을 체크하였으니 제거하여 재생성하고, 모델을 돌려봄

index_max_acc = acc_list.index(max(acc_list))
print(index_max_acc)    # 9
drop_list = np.where(model.best_estimator_.feature_importances_ < th_list[index_max_acc])
print(drop_list)        #(array([14, 18, 20, 21, 28, 32, 38, 41, 49], dtype=int64),)


# x = datasets.drop(['quality'], axis =1)
# y = datasets['quality']




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

# ======================================================================
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.869392356479609
# select_Accuracy :  0.8758465788318718
# select_Accuracy :  0.8741598753904805
# select_Accuracy :  0.8751753397072365
# select_Accuracy :  0.8749860158515701
# select_Accuracy :  0.8761477758749774
# select_Accuracy :  0.8721805805357865
# select_Accuracy :  0.873927523385799
# select_Accuracy :  0.8738414670877688
# select_Accuracy :  0.8751839453370395
# select_Accuracy :  0.8742889598375257
# select_Accuracy :  0.8717675103052417
# select_Accuracy :  0.8729981153670732
# select_Accuracy :  0.8713458344448939
# select_Accuracy :  0.8701152293830624
# select_Accuracy :  0.8706659896904555
# select_Accuracy :  0.8607609097871828
# select_Accuracy :  0.858893488119928
# select_Accuracy :  0.8605629803017134
# select_Accuracy :  0.8606404309699406
# select_Accuracy :  0.8212180408423191
# select_Accuracy :  0.8190063939829436
# select_Accuracy :  0.8170270991282497
# select_Accuracy :  0.7688441778611568
# select_Accuracy :  0.768448318890218
# select_Accuracy :  0.7662366720308426
# select_Accuracy :  0.7414696694577593
# select_Accuracy :  0.7411254442656385
# select_Accuracy :  0.7400841630594736
# select_Accuracy :  0.7376057416762045
# select_Accuracy :  0.7335008562601654
# select_Accuracy :  0.7123568238341523
# select_Accuracy :  0.7068406151304183
# select_Accuracy :  0.706212404154798
# select_Accuracy :  0.7048355033863153
# select_Accuracy :  0.7002314914417012
# select_Accuracy :  0.6998356324707624
# select_Accuracy :  0.6937772690894383
# select_Accuracy :  0.6919959037202138
# select_Accuracy :  0.6849995266903608
# select_Accuracy :  0.684070118671635
# select_Accuracy :  0.6777621920260234
# select_Accuracy :  0.6736659122397872

# ================================= 모델 수정 후
# Score :  0.8750548608899942
# acc :  0.8750548608899942
# ==============================================