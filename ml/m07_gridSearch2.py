# train훈련해서 test로 평가하니까

from tkinter import Grid
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)


# 2.  모델구성
# model =GridSearchCV( SVC(), parameters, cv=kfold , verbose=1, refit=True )
model = SVC(C=1, kernel='linear', degree=3)

# 3. fit
model.fit(x_train, y_train)

# 4. 평가, 예측
print("Model Score : ", model.score(x_test, y_test))#있을 것

y_predict = model.predict(x_test)
print("Accuracy Score : ", accuracy_score(y_test, y_predict))


# gridSearch1.py의 내용과 똑같이 나오는 걸 알 수 있다.

# Model Score :  0.9666666666666667
# Accuracy Score :  0.9666666666666667