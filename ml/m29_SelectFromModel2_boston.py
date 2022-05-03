# 보스톤 만들기 ㄱㄱㄱㄱ
# 중요도 낮은 피처 줄인다음에,
# 다시 모델해서 결과 비교

from xgboost import XGBRegressor
from sklearn.datasets import load_boston


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Normalizer, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
#1. 데이터
# ※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★
# x랑 y바로 분리해주는 return_X_y !!!!!!!!!!!!!!!!
# ※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★

x, y = load_boston(return_X_y=True)
print(x.shape, y.shape)

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)    #, stratify=y) regress

# Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBRegressor(n_jobs=-1)

# 3. 훈련
model.fit( x_train, y_train )

# 4. 평가
score = model.score( x_test, y_test )
print('model.score : ', score)
# model.score :  0.9221188601856797

# 중요도 상관없이 피처 임포턴스
print(model.feature_importances_)
# [0.01447933 0.00363372 0.01479118 0.00134153 0.06949984 0.30128664
#  0.01220458 0.05182539 0.0175432  0.03041654 0.04246344 0.01203114
#  0.4284835 ]

# 컬럼 순서 중요도 순으로 sort하려한다!
print(np.sort(model.feature_importances_))
# [0.00134153 0.00363372 0.01203114 0.01220458 0.01447933 0.01479118
#  0.0175432  0.03041654 0.04246344 0.05182539 0.06949984 0.30128664
#  0.4284835 ]

aaa= np.sort(model.feature_importances_)

print("======================================================================")

# for thresh in aaa:
#     selection = SelectFromModel(model, threshold=thresh, prefit = True )
#     select_x_train  = selection.transform(x_train)
#     select_x_test  = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model =   XGBRegressor(n_jobs=-1)
#     selection_model.fit(select_x_train, y_train)
    
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)
    
#     print("Thresh=%.3f, n=%d, R2: %.2f%%"
#           %(thresh,select_x_train.shape[1], score*100))


      
      
#############################################################################
# 실습 # 셀렉션모델 기반으로 재모델링해서 비교 ㄱㄱ
#############################################################################


# ㄱㄱ
# 이렇게하는거 맞나??ㅠㅠㅠ
# print("======================================================================")
# for i in range(5):
#     # 3. 훈련
#     selection_model[i].fit(x_train, y_train)

#     # 4. 평가, 예측
#     result = selection_model[i].score(x_train, y_train)
#     feature_importances_ = selection_model[i].feature_importances_

#     from sklearn.metrics import accuracy_score
#     y_predict = selection_model[i].predict(x_test)
#     acc = accuracy_score(y_test, y_predict)
#     print("result",result)
#     print("accuracy-score : ", acc)
#     print("feature_importances",feature_importances_)
#     print('r2_score : ', r2_score(y_test, y_predict))
    
# # # 4.1. 예측
# # y_predict = model.predict(x_test)
# # print('r2_score : ', r2_score(y_test, y_predict))


# 소담형님 코드# 소담형님 코드# 소담형님 코드# 소담형님 코드# 소담형님 코드# 소담형님 코드
r2_list = []
th_list = []

for thresh in aaa:    
    selection = SelectFromModel(model, threshold=thresh, prefit= True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    # print(select_x_train.shape,select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs = -1)
    #3 훈련
    selection_model.fit(select_x_train, y_train)

    #4 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)

    select_r2 = r2_score(y_test, select_y_pred)
    # print("select_Score : ", score)
    print("select_R2 : ", select_r2)    
    # print("Thresh = %.3f, n=%d, R2 :%2f%%" %(thresh,select_x_train.shape[1],select_r2*100))
    r2_list.append(select_r2)
    th_list.append(thresh)
    

index_max_acc = r2_list.index(max(r2_list))
print(index_max_acc)
drop_list = np.where(model.feature_importances_ < th_list[index_max_acc])
print(drop_list)
x,y = load_boston(return_X_y=True)
x = np.delete(x,drop_list,axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66
         #, stratify= y
         )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 모델
model = XGBRegressor(n_jobs = -1)
#3 훈련
model.fit(x_train, y_train)

#4 평가 예측
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print('================================= 수정 후')
r2 = r2_score(y_test, y_pred)
print("Score : ", score)
print("r2 : ", r2)
print('=================================')