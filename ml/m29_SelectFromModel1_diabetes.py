# 당뇨병 데이터 만들기 ㄱㄱㄱㄱ
# 중요도 낮은 피처 줄인 다음에,
# 다시 모델해서 결과 비교

# 성능 개선하여 0.51 이상 도출할 것

from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
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
x, y = load_diabetes(return_X_y=True)
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

print(model.feature_importances_)
# 컬럼 순서 중요도 순으로 sort하고싶어 !
print(np.sort(model.feature_importances_))
aaa= np.sort(model.feature_importances_)

      
# 4.1. 예측
y_predict = model.predict(x_test)
print('r2_score : ', r2_score(y_test, y_predict))

# sort한거
# [0.02593721 0.03284872 0.03821949 0.04788679 0.05547739 0.06321319
#  0.06597802 0.07382318 0.19681741 0.39979857]

# sort안한거
# [0.02593721 0.03821949 0.19681741 0.06321319 0.04788679 0.05547739
#  0.07382318 0.03284872 0.39979857 0.06597802]

print("======================================================================")
# feature importance로 중요한 것 솎아 내는 과정

# 0.02~이상
# 모델링서 결과치 뽑고
# 2번째에서는 0.03이 들어가고
# 0.03이 threshold로 들어가고. 0.03.이상인놈들을 다시 정하고
# 0.03에 대해 값을가지고 있고 총 10개의 xtrain컬럼에서 0.03이상인 애들은 9개니까 두번째 포문에선 none, 9 이다

# 첫번짼 다돌고 두번짠 가잔약한놈 빼고 모델링 돌리고 결과출력하고 세번째는 0.03을 xgbreggresor로 받아서
# transform 0.03이상인 놈만 transform해서 뽑아
# 이렇게가다보면 마지막 컬럼만 남게되겠지

# 피처중요도로 중요도 작은놈을 계속 뺐는데 위의 4개 빼고 상위 6개만가지고 했을때 결국 최고 성능 R2:30.09%가 나왔다.

# 피처 몇 개를 빼야 좋은지 몰랐는데, 이제 낮은거부터 빼가면서 최고 성능 찾기가 용이해졌다.

for thresh in aaa:
    selection = SelectFromModel(model, threshold=thresh, prefit = True )
    select_x_train  = selection.transform(x_train)
    select_x_test  = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model =   XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%"
          %(thresh,select_x_train.shape[1], score*100))
      



# selectFromModel돌려보고 난걸 기반으로

# 1,
# datasets = load_diabetes()
# print(datasets.feature_names)#피처이름 보기보기~
# np.delete로 네개 지우기

# 2.