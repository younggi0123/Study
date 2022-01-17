# from re import M
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

import warnings#예외처리를 통해 에러메세지를 지우려 씀
warnings.filterwarnings('ignore')

datasets = load_iris()
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
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# 리스트의 딕셔너리 키 밸류 형태 키=이름, 밸류=값

print("allAlgorithms : ", allAlgorithms)
print( "모델의 개수 : ", len(allAlgorithms) )   # 41

for(name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, "의 정답률 : ", acc) # 이름 출력
    except:
        # continue
        print(name, '은 에러난 놈!!!')

# 오래된 algorithm or version 으로 인한 문제로 결과값이 안나오는 경우가 있음



# 모델의 개수 :  41
# AdaBoostClassifier 의 정답률 :  1.0
# BaggingClassifier 의 정답률 :  1.0
# BernoulliNB 의 정답률 :  0.36666666666666664
# CalibratedClassifierCV 의 정답률 :  0.9666666666666667
# CategoricalNB 은 에러난 놈!!!
# ClassifierChain 은 에러난 놈!!!
# ComplementNB 의 정답률 :  0.7
# DecisionTreeClassifier 의 정답률 :  1.0
# DummyClassifier 의 정답률 :  0.3
# ExtraTreeClassifier 의 정답률 :  1.0
# ExtraTreesClassifier 의 정답률 :  1.0
# GaussianNB 의 정답률 :  1.0
# GaussianProcessClassifier 의 정답률 :  0.9666666666666667
# GradientBoostingClassifier 의 정답률 :  1.0
# HistGradientBoostingClassifier 의 정답률 :  1.0
# KNeighborsClassifier 의 정답률 :  1.0
# LabelPropagation 의 정답률 :  1.0
# LabelSpreading 의 정답률 :  1.0
# LinearDiscriminantAnalysis 의 정답률 :  1.0
# LinearSVC 의 정답률 :  0.9666666666666667
# LogisticRegression 의 정답률 :  0.9666666666666667
# LogisticRegressionCV 의 정답률 :  1.0
# MLPClassifier 의 정답률 :  0.9666666666666667
# MultiOutputClassifier 은 에러난 놈!!!
# MultinomialNB 의 정답률 :  0.6333333333333333
# NearestCentroid 의 정답률 :  0.9666666666666667
# NuSVC 의 정답률 :  1.0
# OneVsOneClassifier 은 에러난 놈!!!
# OneVsRestClassifier 은 에러난 놈!!!
# OutputCodeClassifier 은 에러난 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  1.0
# Perceptron 의 정답률 :  0.8
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
# RadiusNeighborsClassifier 의 정답률 :  0.5666666666666667
# RandomForestClassifier 의 정답률 :  1.0
# RidgeClassifier 의 정답률 :  0.9
# RidgeClassifierCV 의 정답률 :  0.8666666666666667
# SGDClassifier 의 정답률 :  0.9
# SVC 의 정답률 :  1.0
# StackingClassifier 은 에러난 놈!!!
# VotingClassifier 은 에러난 놈!!!