import numpy as np
from sklearn.datasets import load_iris

# Data
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델구성

# feature는 tree계열에만 있어
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# model = DecisionTreeClassifier(max_depth=5)
# model = RandomForestClassifier(max_depth=5)
# model = XGBClassifier(max_depth=5)
model = GradientBoostingClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print(model, "accuracy-score : ", acc)
print(model.feature_importances_)


# Classifier별 결과치

# DecisionTreeClassifier
# accuracy-score :  0.9666666666666667
# [0.         0.0125026  0.03213177 0.95536562] #합이 1이다.
# 자원의효율성 -> 0인 feature을 뺏을때 나머지 세개로 돌렸을때 0.96이 나오는가?

# RandomForestClassifier
# RandomForestClassifier(max_depth=5) accuracy-score :  1.0
# [0.08794234 0.02659081 0.47266114 0.41280571]

# XGBClassifier
# accuracy-score :  0.9
# [0.01835513 0.0256969  0.6204526  0.33549538]

# GradientBoostingClassifier
# GradientBoostingClassifier() accuracy-score :  0.9333333333333333
# [0.0036034  0.01236995 0.26689114 0.71713552]