# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성 후 각 모델별로 돌려서 결과 도출!

# 기존 모델결과와 비교

# 결과 비교
# 1. DecisionTree
# 기존 acc :
# 컬럼삭제 후 acc : 

import numpy as np
from sklearn.datasets import load_boston
from pandas import read_csv

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = read_csv(path+'train.csv')  
# test_file = read_csv(path+'test.csv')
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
# test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']


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


import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    
plot_feature_importances_dataset(model)
plt.show()
