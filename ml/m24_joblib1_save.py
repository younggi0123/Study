from tabnanny import verbose
from sklearn.datasets import load_boston
from sklearn.model_selection import learning_curve, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, accuracy_score
import time
# import warnings
# warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets['target'] #  .target과 같아.
print(x.shape, y.shape)     #(20640, 8) (20640,)

x_train, x_test, y_train,y_test = train_test_split(
    x, y, shuffle=True, random_state=66, train_size=0.8) # stratify 분류모델용임

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBRegressor(
                    n_jobs = -1,
                    n_estimators = 1000,
                    learning_rate = 0.025,
                    max_depth = 4,
                    min_child_weight = 1,
                    subsample = 1,
                    colsample_bytree = 1,
                    reg_alpha = 1,
                    reg_lambda = 0
                    )

# 3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 1,
                eval_set = [ (x_train, y_train), (x_test, y_test) ],    # train은 원래 들어가는게 아닌거알지? 결과치 확인용이야~
                # loss
                eval_metric='rmse',
                early_stopping_rounds=10 # evalmetric기준 earlystopping은 1000번도는 동안 10단위로
            )

end = time.time()
print("걸린시간 : ", end-start)

results = model.score(x_test, y_test)
print("result : ", round(results, 4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", round(r2,4))

print("========================================")
hist = model.evals_result()
print(hist)


# 저장
# import pickle
# path = '../_save/'
# pickle.dump(model, open(path + 'm23_pickle1_save.dat','wb'))

import joblib
path =  '../_save/'
joblib.dump(model, path + 'm24_joblib1_save.dat')