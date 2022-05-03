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

# 이를 찍어 봄으로써 앞부분 train 뒷부분test. test가 계속 줄어들고 있다면 훈련수가 부족한거겠지?
# 그게아니라면 얼리스타핑써야겠지?

# 실습
# Hist 그림으로 그리기 ! !

import matplotlib.pyplot as plt

# plt.hist(hist['validation_0'], histtype='step')
# plt.hist(hist['validation_1'], histtype='step')
# plt.show()


plt.figure(figsize=(9,6))
# dict내의 validation_0의 rmse를 지정함.
plt.plot( hist['validation_0']['rmse'], marker=".", c='red', alpha=0.3, label='train_set' )
plt.plot( hist['validation_1']['rmse'], marker='.', c='blue', alpha =0.3,label='test_set' )
plt.grid() 
plt.title('loss_rmse')
plt.ylabel('loss_rmse')
plt.xlabel('epoch')
plt.legend(loc='upper right') 
plt.show()