import numpy as np
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_wine()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)
n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)
model = SVC()

scores = cross_val_score(model, x, y, cv=kfold)
print(" ACC : ", scores, "\n Cross_Val_Score : ", round(np.mean(scores), 4) )
 
 
#  ACC :  [0.55555556 0.77777778 0.72222222 0.71428571 0.68571429] 
#  Cross_Val_Score :  0.6911