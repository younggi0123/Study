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
import warnings
warnings.filterwarnings('ignore')

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

# load boston data set case
# model = XGBRegressor(
#                     n_jobs = -1,
#                     n_estimators = 2000,
#                     learning_rate = 0.0548,    # 0.9423
#                     max_depth = 3           # 0.9372
#                     )
model = XGBRegressor(
                    n_jobs = -1,
                    n_estimators = 2000,
                    learning_rate = 0.025,
                    max_depth = 4,
                    min_child_weight = 1,
                    subsample = 1,
                    colsample_bytree = 1,
                    reg_alpha = 1,  # 규제 L1
                    reg_lambda = 0  # 규제 L2
                    )


# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()
print("걸린시간 : ", end-start)

results = model.score(x_test, y_test)
print("result : ", round(results, 4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", round(r2,4))

print("========================================")
# hist = model.evals_result()
# print(hist)   # error : xgboost.core.XGBoostError: No evaluation result, `eval_set` is not used during training.