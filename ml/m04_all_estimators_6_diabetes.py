# from re import M
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_diabetes

import warnings#예외처리를 통해 에러메세지를 지우려 씀
warnings.filterwarnings('ignore')

datasets = load_diabetes()
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

# 모델의 개수 :  54
# ARDRegression 의 r2값 :  0.4670336652870781
# AdaBoostRegressor 의 r2값 :  0.42623401535928274
# BaggingRegressor 의 r2값 :  0.4542917618414236
# BayesianRidge 의 r2값 :  0.45770277915190327
# CCA 의 r2값 :  0.4381070029563098
# DecisionTreeRegressor 의 r2값 :  0.19693003922637642
# DummyRegressor 의 r2값 :  -0.011962984778542296
# ElasticNet 의 r2값 :  0.13082214338749865
# ElasticNetCV 의 r2값 :  0.4607369144104332
# ExtraTreeRegressor 의 r2값 :  -0.0064841233493444506
# ExtraTreesRegressor 의 r2값 :  0.4606857195902653
# GammaRegressor 의 r2값 :  0.08029031908102047
# GaussianProcessRegressor 의 r2값 :  -17.734682101728374
# GradientBoostingRegressor 의 r2값 :  0.45104521134782183
# HistGradientBoostingRegressor 의 r2값 :  0.37519090810376543
# HuberRegressor 의 r2값 :  0.4472595383583047
# IsotonicRegression 은 에러난 놈!!!
# KNeighborsRegressor 의 r2값 :  0.4503424728105596
# KernelRidge 의 r2값 :  0.45603293351121643
# Lars 의 r2값 :  -1.203367750585898
# LarsCV 의 r2값 :  0.4722505318607023
# Lasso 의 r2값 :  0.46972020530847247
# LassoCV 의 r2값 :  0.47276199464335056
# LassoLars 의 r2값 :  0.37890235603594236
# LassoLarsCV 의 r2값 :  0.4716071663733019
# LassoLarsIC 의 r2값 :  0.4712510777170408
# LinearRegression 의 r2값 :  0.45260660216173787
# LinearSVR 의 r2값 :  0.23859848814982465
# MLPRegressor 의 r2값 :  -0.4954096589003891
# MultiOutputRegressor 은 에러난 놈!!!
# MultiTaskElasticNet 은 에러난 놈!!!
# MultiTaskElasticNetCV 은 에러난 놈!!!
# MultiTaskLasso 은 에러난 놈!!!
# MultiTaskLassoCV 은 에러난 놈!!!
# NuSVR 의 r2값 :  0.15191974468247293
# OrthogonalMatchingPursuit 의 r2값 :  0.23335039815872138
# OrthogonalMatchingPursuitCV 의 r2값 :  0.46936551659312953
# PLSCanonical 의 r2값 :  -1.5249671559732398
# PLSRegression 의 r2값 :  0.4413362659940103
# PassiveAggressiveRegressor 의 r2값 :  0.42367427048324047
# PoissonRegressor 의 r2값 :  0.4441692945832215
# RANSACRegressor 의 r2값 :  0.12314934159462254
# RadiusNeighborsRegressor 의 r2값 :  0.1512475693384715
# RandomForestRegressor 의 r2값 :  0.411797465523823
# RegressorChain 은 에러난 놈!!!
# Ridge 의 r2값 :  0.45921222867719014
# RidgeCV 의 r2값 :  0.4552971481969251
# SGDRegressor 의 r2값 :  0.4593411984202407
# SVR 의 r2값 :  0.15875529246365316
# StackingRegressor 은 에러난 놈!!!
# TheilSenRegressor 의 r2값 :  0.45129901220722124
# TransformedTargetRegressor 의 r2값 :  0.45260660216173787
# TweedieRegressor 의 r2값 :  0.07699640230917337
# VotingRegressor 은 에러난 놈!!!