from sklearn.datasets import load_wine
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
# import warnings
# warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_wine()
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

model = XGBClassifier(
                    n_jobs = -1,
                    n_estimators = 2000,
                    learning_rate = 0.039,
                    max_depth = 7,      #maxdepth올렸을때 성능올랐음
                    min_child_weight = 1,
                    subsample = 1,
                    colsample_bytree = 1,
                    reg_alpha = 1,  # 규제 L1
                    reg_lambda = 0  # 규제 L2
                    )

# default값은 무엇인가?



# 3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set = [ (x_test, y_test) ],
          eval_metric='mlogloss',            # rmse, mae, logloss, error
          
          )
            # evaluate가 무엇인지 명시해줘야 verbose를 표시할 수 있다.(즉, eval_set을!)
           # 다중분류 multi -> mlogloss가능 (rmse말고)
end = time.time()
print("걸린시간 : ", end-start)

results = model.score(x_test, y_test)
print("result : ", round(results, 4))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy score: ", round(acc,4))

print("========================================")
# hist = model.evals_result()
# print(hist)   # error : xgboost.core.XGBoostError: No evaluation result, `eval_set` is not used during training.