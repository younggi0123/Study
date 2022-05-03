import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from xgboost import XGBClassifier

from sklearn.model_selection import KFold

# read_csv takes a sep param, in your case just pass sep=';' like so:
# data = read_csv(csv_path, sep=';')

# 데이터 불러오기
path = "../_data/kaggle/wine/"

# read_csv takes a sep param, in your case just pass sep=';' like so:
# data = read_csv(csv_path, sep=';')
train = pd.read_csv(path+"winequality-white.csv",sep=';')

y = train['quality']
x = train.drop(['quality'], axis =1)
print(x)




# # [[ 아웃라이어 함수 적용 ]]
# def outliers(data_out):
#     quantile_1, q2, quantile_3 = np.percentile(data_out, [25,50,75])
#     print("1사분위 : ", quantile_1)
#     print("q2 : ", q2)
#     print("3사분위 : ", quantile_3)
#     iqr = quantile_3 - quantile_1
#     print("iqr : ", iqr)
#     lower_bound = quantile_1 - (iqr * 1.5)
#     upper_bound = quantile_3 + (iqr * 1.5)
#     return np.where((data_out>upper_bound) |        #  이 줄 또는( | )
#                     (data_out<lower_bound))         #  아랫줄일 경우 반환

# print( outliers(x) )

# outliers_loc = outliers(x)
# print("이상치의 위치 : ", outliers_loc)















print(x.shape, y.shape)
x_train, x_test, y_train,y_test = train_test_split(
    x, y, shuffle=True, random_state=42, train_size=0.8) # stratify 분류모델용임

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

model = XGBClassifier(n_jobs = -1,
                      n_estimators = 5000,
                      learning_rate = 0.05,
                      max_depth = 5,
                      min_child_weight = 1,
                      subsample = 1,
                      colsample_bytree = 1,
                      reg_alpha = 1,
                      reg_lambda = 0,
                      cv = cv
                    )

# 3. 훈련
import time
start = time.time()
model.fit(x_train, y_train, verbose = 1,
                eval_set = [ (x_test, y_test) ],    # train은 원래 들어가는게 아닌거알지? 결과치 확인용이야~
                eval_metric='mlogloss',
                # early_stopping_rounds=1000 # evalmetric기준 earlystopping은 1000번도는 동안 10단위로
            )

end = time.time()
print("걸린시간 : ", end-start)


# 4. 평가, 예측
# result = model.score(x_train, y_train)
# result = model.score(x_test, y_test)
# print(result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print(model, "accuracy-score : ", acc)