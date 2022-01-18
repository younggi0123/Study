import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)
n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)
model = SVC()

scores = cross_val_score(model, x, y, cv=kfold)
print(" ACC : ", scores, "\n Cross_Val_Score : ", round(np.mean(scores), 4) )




