# from re import M
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_diabetes
from pandas import read_csv
from sklearn.metrics import r2_score


#1. 데이터 
path = '../_data/kaggle/bike/'   
train = read_csv(path+'train.csv')  
print(train.shape)      # (10886, 12)
test_file = read_csv(path+'test.csv')
print(test_file.shape)    # (6493, 9)
submit_file = read_csv(path+ 'sampleSubmission.csv')
print(submit_file.shape)     # (6493, 2)
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']

import warnings#예외처리를 통해 에러메세지를 지우려 씀
warnings.filterwarnings('ignore')


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
# ARDRegression 의 r2값 :  0.27553327314293163
# AdaBoostRegressor 의 r2값 :  0.25099360529976644
# BaggingRegressor 의 r2값 :  0.2369811126938095
# BayesianRidge 의 r2값 :  0.2756695287304274
# CCA 의 r2값 :  -0.06978595926405573
# DecisionTreeRegressor 의 r2값 :  -0.13328795261926252
# DummyRegressor 의 r2값 :  -8.532574979902563e-08
# ElasticNet 의 r2값 :  0.06035250425665917
# ElasticNetCV 의 r2값 :  0.2607490055621796
# ExtraTreeRegressor 의 r2값 :  -0.12558265955963566
# ExtraTreesRegressor 의 r2값 :  0.2134045652995583
# GammaRegressor 의 r2값 :  0.03500071239049418
# GaussianProcessRegressor 의 r2값 :  -1.757621821646567
# GradientBoostingRegressor 의 r2값 :  0.34568590588540093
# HistGradientBoostingRegressor 의 r2값 :  0.37270207646202547
# HuberRegressor 의 r2값 :  0.24700257405395598
# IsotonicRegression 은 에러난 놈!!!
# KNeighborsRegressor 의 r2값 :  0.3137506639116687
# KernelRidge 의 r2값 :  0.2504164479826685
# Lars 의 r2값 :  0.2757033009386365
# LarsCV 의 r2값 :  0.2751951234147688
# Lasso 의 r2값 :  0.27003063411600847
# LassoCV 의 r2값 :  0.2755368998428025
# LassoLars 의 r2값 :  -8.532574979902563e-08
# LassoLarsCV 의 r2값 :  0.2751951234147688
# LassoLarsIC 의 r2값 :  0.27567298983778266
# LinearRegression 의 r2값 :  0.2757033009386365
# LinearSVR 의 r2값 :  0.1965725618292995
# MLPRegressor 의 r2값 :  0.28046439843689885
# MultiOutputRegressor 은 에러난 놈!!!
# MultiTaskElasticNet 은 에러난 놈!!!
# MultiTaskElasticNetCV 은 에러난 놈!!!
# MultiTaskLasso 은 에러난 놈!!!
# MultiTaskLassoCV 은 에러난 놈!!!
# NuSVR 의 r2값 :  0.21411224437655196
# OrthogonalMatchingPursuit 의 r2값 :  0.16926571255409573
# OrthogonalMatchingPursuitCV 의 r2값 :  0.2740905595558095
# PLSCanonical 의 r2값 :  -0.5136954427382536
# PLSRegression 의 r2값 :  0.269035405681765
# PassiveAggressiveRegressor 의 r2값 :  0.21717766591313092
# PoissonRegressor 의 r2값 :  0.2721648819371608
# RANSACRegressor 의 r2값 :  -0.021915965797421277
# RadiusNeighborsRegressor 의 r2값 :  0.06155703383586175
# RandomForestRegressor 의 r2값 :  0.30049589009937683
# RegressorChain 은 에러난 놈!!!
# Ridge 의 r2값 :  0.2756770331045797
# RidgeCV 의 r2값 :  0.2756770331045971
# SGDRegressor 의 r2값 :  0.27422469671539496
# SVR 의 r2값 :  0.2091892482842721
# StackingRegressor 은 에러난 놈!!!
# TheilSenRegressor 의 r2값 :  0.262726479306606
# TransformedTargetRegressor 의 r2값 :  0.2757033009386365
# TweedieRegressor 의 r2값 :  0.03527268923393789
# VotingRegressor 은 에러난 놈!!!