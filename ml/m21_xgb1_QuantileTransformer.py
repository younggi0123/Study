from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.model_selection import learning_curve, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
# 최근 나온 전처리 기법 = 알아서 찾아. (이상치에 자유)
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, accuracy_score
import time
# 1. 데이터
# datasets = fetch_california_housing()
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
# model = XGBRegressor()                    # result :  0.843390038548427

# fetch_california_housing data set case
# model = XGBRegressor(   # n_estimators are just like epochs
#                     n_jobs = -1,
#                     # n_estimators = 200,     # result :  0.847329165954032
#                     n_estimators = 2000,     # result :  0.8566291699938181
#                     learning_rate = 0.1,    # result :  0.8532492079366552
#                     max_depth = 5           # result :  0.8588776275089584
#                     )


# load boston data set case
model = XGBRegressor(
                    n_jobs = -1,
                    n_estimators = 2000,     # result :  0.9360605688665197
                    learning_rate = 0.1,
                    max_depth = 3
                    )



# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()
print("걸린시간 : ", end-start)

results = model.score(x_test, y_test)
print("result : ", results)
# result :  0.843390038548427

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)
# r2 :  0.843390038548427

print("========================================")
# hist = model.evals_result()
# print(hist)   # error : xgboost.core.XGBoostError: No evaluation result, `eval_set` is not used during training.