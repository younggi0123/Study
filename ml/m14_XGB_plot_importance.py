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
model = XGBClassifier(max_depth=5)
# model = GradientBoostingClassifier()

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

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)
    
# plot_feature_importances_dataset(model)
# plt.show()


from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()