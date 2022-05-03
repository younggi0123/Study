# Subplot 이용해서 4개의 모델을 한 화면에 그래프로 ㄱㄱ
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

model1 = DecisionTreeClassifier(max_depth=5)
model2 = RandomForestClassifier(max_depth=5)
model3 = XGBClassifier()
model4 = GradientBoostingClassifier()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

# 4. 평가, 예측
result1 = model1.score(x_train, y_train)
result2 = model2.score(x_train, y_train)
result3 = model3.score(x_train, y_train)
result4 = model4.score(x_train, y_train)

# enumerate

# 4가지 모델에 대한 결과치들 아래에 각각 출력
print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)



import matplotlib.pyplot as plt
def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    
plt.subplot(2,2,1 )                # nrows=2, ncols=2, index=1
plot_feature_importances_dataset(model1)

plt.subplot(2,2,2 )                # nrows=2, ncols=2, index=1
plot_feature_importances_dataset(model2)

plt.subplot(2,2,3 )                # nrows=2, ncols=2, index=1
plot_feature_importances_dataset(model3)

plt.subplot(2,2,4 )                # nrows=2, ncols=2, index=1
plot_feature_importances_dataset(model4)

# plt.show()

#     model_list[i].fit(x_train, y_train)

#     # 4. 평가, 예측
#     result = model_list[i].score(x_train, y_train)
#     feature_importances_ = model_list[i].feature_importances_

#     from sklearn.metrics import accuracy_score
#     y_predict = model_list[i].predict(x_test)
#     acc = accuracy_score(y_test, y_predict)
#     print("result",result)
#     print("accuracy-score : ", acc)
#     print("feature_importances",feature_importances_)
#     plot_feature_importances_dataset(model_list[i])
#     plt.ylabel(model_name[i])



# # plot_feature_importances_dataset(model)
# plt.show()
