from distutils.archive_util import make_archive
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sympy import Line

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(datasets.feature_names)
print(datasets.DESCR)

print(x.shape, y.shape) # (581012, 54) (581012,)

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# x_train, x_test, y_train, y_test  =  train_test_split(
#                     x, y, test_size=0.1, random_state=66)

# model = LinearRegression()
# # model = make_pipeline(StandardScaler(), LinearRegression() )

# model.fit(x_train, y_train)

# print(model.score(x_test, y_test))
# # 0.3231827383051925        # linear regression
# # 0.3231860467723632        # pipeline

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2') # scoring : 어떤 평가지표?
# print(scores)
# Linear Regression : [0.32356477 0.3173615  0.31295288 0.31752544 0.31760723 0.31954103 0.31989583]
# make pipeline : [0.32355793 0.31736138 0.31295749 0.31752546 0.31760747 0.31954592 0.31989767]










###################################################### Polynomial Features 후 ######################################################
from sklearn.preprocessing import PolynomialFeatures # 일종의 scaling임을 알 수있으며 y가 없어도 됨을 알 수 있다.
pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
print(xp.shape)         # (581012, 1540)

# 폴리노미얼 : 컬럼생성할 건데 약간 증폭의 개념
# y = w1x1 + w2x2 + b 가있다면
# 아얘 새로운 데이터 셋을 만들것이고 컬럼의 양을 늘린다. 어떤애는 제곱하는 등으로.
# 위와는 다른 컬럼이 됨, y = x0 + w1x1^2 + w2x1x2 + w3x2^2 + b  (2차함수 형태로 만들기 위해 원래있던 애들을 제곱한 개념)
# 데이터의 구조가 ~~


x_train, x_test, y_train, y_test  =  train_test_split(
                    xp, y, test_size=0.1, random_state=66)

model = LinearRegression()
# model = make_pipeline(StandardScaler(), LinearRegression() )

model.fit(x_train, y_train)

print(model.score(x_test, y_test))
# 0.4854930214665447        # LinearRegression
# 0.4854729169563706        # Pipeline

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2') # scoring : 어떤 평가지표?
print(scores)
# LinearRegression : [ 0.49067729  0.48453863  0.47969478  0.47876661 -7.08505925  0.48891913  0.48355037]
# Pipeline : [ 4.90679809e-01  4.84539849e-01 -4.07649670e+21  4.78751733e-01  -4.49871856e+21  4.88924313e-01 -5.35154595e+21]


