# from re import M
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_boston

import warnings#예외처리를 통해 에러메세지를 지우려 씀
warnings.filterwarnings('ignore')

datasets = load_boston()
x= datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

print("allAlgorithms : ", allAlgorithms)
print("모델의 개수 : ",len(allAlgorithms))   # 모델의 갯수 :  41(classifier) / 54(regressor)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 r2값 : ', r2)
    except:
        # continue
        print(name, '은 에러난 놈!!!')


# 오래된 algorithm or version 으로 인한 문제로 결과값이 안나오는 경우가 있음



# 모델의 갯수 :  54
# ARDRegression 의 r2값 :  0.6631474329953632
# AdaBoostRegressor 의 r2값 :  0.8317742656069406
# BaggingRegressor 의 r2값 :  0.8700408907526361
# BayesianRidge 의 r2값 :  0.6712510472712032
# CCA 의 r2값 :  0.5198775240037121
# DecisionTreeRegressor 의 r2값 :  0.68263429966378
# DummyRegressor 의 r2값 :  -0.023340500652033302
# ElasticNet 의 r2값 :  0.1596567593503213
# ElasticNetCV 의 r2값 :  0.67389268155133
# ExtraTreeRegressor 의 r2값 :  0.7184054706175914
# ExtraTreesRegressor 의 r2값 :  0.8623305773525454
# GammaRegressor 의 r2값 :  0.2019016158602781
# GaussianProcessRegressor 의 r2값 :  -0.41052624661958825
# GradientBoostingRegressor 의 r2값 :  0.9161294999447543
# HistGradientBoostingRegressor 의 r2값 :  0.8617126531575241
# HuberRegressor 의 r2값 :  0.6147327285078037
# KNeighborsRegressor 의 r2값 :  0.7039265754739785
# KernelRidge 의 r2값 :  0.5263239302949906
# Lars 의 r2값 :  0.6687594935356316
# LarsCV 의 r2값 :  0.6646815121672811
# Lasso 의 r2값 :  0.2573921442545194
# LassoCV 의 r2값 :  0.6688640223695403
# LassoLars 의 r2값 :  -0.023340500652033302
# LassoLarsCV 의 r2값 :  0.6687594935356316
# LassoLarsIC 의 r2값 :  0.6328461470661211
# LinearRegression 의 r2값 :  0.668759493535632
# LinearSVR 의 r2값 :  0.6059981867507432
# MLPRegressor 의 r2값 :  0.3948437657211966
# NuSVR 의 r2값 :  0.587810983300225
# OrthogonalMatchingPursuit 의 r2값 :  0.5429180422970386
# OrthogonalMatchingPursuitCV 의 r2값 :  0.606107808750429
# PLSCanonical 의 r2값 :  -2.8693001272936116
# PLSRegression 의 r2값 :  0.6137785973148175
# PassiveAggressiveRegressor 의 r2값 :  0.6443918735702456
# PoissonRegressor 의 r2값 :  0.6260264594204782
# RANSACRegressor 의 r2값 :  0.5480457992174927
# RadiusNeighborsRegressor 의 r2값 :  0.2900138666604456
# RandomForestRegressor 의 r2값 :  0.8820721468993821
# Ridge 의 r2값 :  0.67641003654236
# RidgeCV 의 r2값 :  0.6700309977617632
# SGDRegressor 의 r2값 :  0.6343001259441776
# SVR 의 r2값 :  0.6194523099637248
# TheilSenRegressor 의 r2값 :  0.6219372671305725
# TransformedTargetRegressor 의 r2값 :  0.668759493535632
# TweedieRegressor 의 r2값 :  0.1913674200549934