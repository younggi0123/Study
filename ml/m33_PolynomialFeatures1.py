from distutils.archive_util import make_archive
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sympy import Line

datasets = load_boston()                
# datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(datasets.feature_names)
print(datasets.DESCR)

# print(x.shape, y.shape)                 # fetch_california_housing : (20640, 8) (20640,)
                                          # load_boston : (506,13)(506,)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

x_train, x_test, y_train, y_test  =  train_test_split(
                    x, y, test_size=0.1, random_state=66)

# model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression() )

model.fit(x_train, y_train)

print(model.score(x_test, y_test))
# 0.7795056314949791        # 보통
# 0.77950563149498          # pipeline

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2') # scoring : 어떤 평가지표?
print(scores)
# [0.83841344 0.81105121 0.65897081 0.63406181 0.71122933 0.51831124 0.73634677]

# import sklearn
# print(sklearn.metrics.SCORERS.keys())   #scoring의 지표들을 확인











###################################################### Polynomial Features 후 ######################################################
from sklearn.preprocessing import PolynomialFeatures # 일종의 scaling임을 알 수있으며 y가 없어도 됨을 알 수 있다.
pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
print(xp.shape)         # 보스톤 기준 (506, 105)

# 폴리노미얼 : 컬럼생성할 건데 약간 증폭의 개념
# y = w1x1 + w2x2 + b 가있다면
# 아얘 새로운 데이터 셋을 만들것이고 컬럼의 양을 늘린다. 어떤애는 제곱하는 등으로.
# 위와는 다른 컬럼이 됨, y = x0 + w1x1^2 + w2x1x2 + w3x2^2 + b  (2차함수 형태로 만들기 위해 원래있던 애들을 제곱한 개념)
# 데이터의 구조가 ~~


x_train, x_test, y_train, y_test  =  train_test_split(
                    xp, y, test_size=0.1, random_state=66)

# model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression() )

model.fit(x_train, y_train)

print(model.score(x_test, y_test))
# 0.9382991502184762          # pipeline

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2') # scoring : 어떤 평가지표?
print(scores)
# [0.73130288 0.85846952 0.72049694 0.77758685 0.88984214 0.65914223 0.86252149]

