# VotingClassifier사용해 보기 for문 안돌려도 한번에 가능해.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

n_splits = 5
# shuffle 재구성한 다음에 자르겠다
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = SVC()

# cross_val_score 뿐 아니라 다양하게 사용함
scores = cross_val_score(model, x, y, cv=kfold)# crossval만큼 훈련시키겠다(5번)
print(" ACC : ", scores, "\n Cross_Val_Score : ", round(np.mean(scores), 4) )


#  ACC :  [0.         0.01123596 0.01136364 0.         0.        ] 
#  Cross_Val_Score :  0.0045