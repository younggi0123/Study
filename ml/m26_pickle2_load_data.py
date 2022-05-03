from tabnanny import verbose
from sklearn.datasets import fetch_covtype
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
# datasets = fetch_covtype()
# x = datasets.data
# y = datasets['target']
# print(x.shape, y.shape)

import pickle
path = '../_save/'
datasets = pickle.load(open(path + 'm26_pickle1_save_datasets.py', 'rb'))   # read binary
x = datasets.data
y = datasets['target']
# 넘파이 저장안하고 피클로 저장하는 이유?
# 넘파이가 훨빠르지만 피클은 컬럼 이름 및 인덱스가 같이 저장 된다.
# 즉, 데이터프레임의 특성 그대로 저장이 가능하단말
# 넘파이로 바꿔저장하면 인덱스랑 칼럼을 넣을 수 없어.
# 대회에서 데이터받았는데 점데이터다?? 피클인지 잡립인지도 고려해봐야해( 데이터셋을 폐쇄망으로 받을 수도 있어. )

x_train, x_test, y_train,y_test = train_test_split(
    x, y, shuffle=True, random_state=66, train_size=0.8) # stratify 분류모델용임


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(
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
                eval_metric='mlogloss',     # fetch_covtype은 7개로 multi니까.
                early_stopping_rounds=10 # evalmetric기준 earlystopping은 1000번도는 동안 10단위로
            )

end = time.time()
print("걸린시간 : ", end-start)

results = model.score(x_test, y_test)
print("result : ", round(results, 4))

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print("r2 : ", round(acc,4))




print("========================================")
hist = model.evals_result()
print(hist)


# 저장
# import pickle
# path = '../_save/'
# pickle.dump(model, open(path + 'm26_pickle1_save_model.dat','wb'))