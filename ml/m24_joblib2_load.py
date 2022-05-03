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

# ● 모델 불러오기 // 2. 모델 3. 훈련
# 데이터는 따로 저장할 수 있지만 비효율적이겠지?
import joblib
path= '../_save/'
model = joblib.load( path + 'm24_joblib1_save.dat' )

# 4. 평가
results = model.score(x_test, y_test)
print("result : ", round(results, 4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", round(r2,4))

print("========================================")
hist = model.evals_result()
print(hist)

# result :  0.9452
# r2 :  0.9452

# csv파일이 생각보다 크니까
# 피클 혹은 잡립 쓰면 바이너리 상태로 저장되어 데이터 크기가 작아지니까 땡기는게 쉬워져
