# VotingClassifier사용해 보기 for문 안돌려도 한번에 가능해.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

# 예를들면 5로 트레인테스트 스플릿을 나누면
# 5번의 학습을 진행하는동안 테스트와 트레인을 바꿔학습하면서
# 주어진 데이터를 다 사용하겠다는 말.
# 랜덤스테이트를 지정해줘야 랜덤값이 고정된다.


n_splits = 5
# shuffle 재구성한 다음에 자르겠다
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = SVC()

# cross_val_score 뿐 아니라 다양하게 사용함
scores = cross_val_score(model, x, y, cv=kfold)# crossval만큼 훈련시키겠다(5번)
print(" ACC : ", scores, "\n Cross_Val_Score : ", round(np.mean(scores), 4) )

# for문 아니고 알아서 5번 돌아간다.

# Print 첫번째 훈련  두번째 훈련  세번째 훈련 네번째훈련 다섯번째 훈련
# ACC :  [0.96666667 0.96666667   1.          0.93333333 0.96666667]