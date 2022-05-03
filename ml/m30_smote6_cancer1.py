# SMOTE 넣어서 Breast Cancer Data 만들기
# 데이터 값 넣은 것과 안 넣은 것의 차이를 비교해본다.

# 그냥 증폭해서 성능비교 
# 총 7개 라벨을 증폭.
# x_train을 증폭하고 x_test는 철저히 평가해석만 하여 성능비교

# ※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★ 데이터 증폭 수업 ※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (569, 30) (569,)
print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

print(np.unique(y, return_counts=True))
# (array([0, 1]),
#  array([212, 357], dtype=int64))
# =>      0이 212개 , 1이 357개

x_train, x_test, y_train, y_test \
    = train_test_split(x, y, train_size=0.75,
    shuffle=True, random_state=66, stratify=y )

#####################################################################################
#                                    ※ 기본 ※
#####################################################################################

# model = XGBClassifier(n_jobs=4)
# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)
# print("model.score : ",
#       round(score,4) )
# y_predict = model.predict(x_test)
# print("accuracy score : ",
#       round(accuracy_score(y_test, y_predict), 4 ) )
# print('f1_score : ',
#       round(f1_score(y_test, y_predict, average='macro'), 4) )
# # print('f1_score : ',f1_score(y_test, y_predict, average='micro'))

# # model.score :  0.951
# # accuracy score :  0.951
# # f1_score :  0.9477




#####################################################################################
#                                 ※ 데이터 증폭 ※
#####################################################################################


smote = SMOTE(random_state=66)

x_train, y_train = smote.fit_resample(x_train, y_train) # fit resample에 xtrain ytrain 넣은 것을 반환해줘야

#                                    S M O T E

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ",
      round(score,4) )
y_predict = model.predict(x_test)
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ) )
print('f1_score : ',
      round(f1_score(y_test, y_predict, average='macro'), 4) )
# print('f1_score : ',f1_score(y_test, y_predict, average='micro'))


# model.score :  0.951
# accuracy score :  0.951
# f1_score :  0.9477