# ML 03_1~  iris를 사용한 실습 진행

from distutils.log import Log
import numpy as np
from sklearn.datasets import load_wine

# Data
datasets = load_wine()

x = datasets.data
y = datasets.target

# 기본적 레거시한 머신러닝은 카테고리컬 조차 필요가 없다.
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)      #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# LogisticRegression
#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression
#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression
#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression
#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression
#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression#################이름은 이따군데 절대 분류임# LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


pc = Perceptron()
lsvc = LinearSVC()
svc = SVC()
knc = KNeighborsClassifier()
lr = LogisticRegression()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()

def banbok(model):
    #3. 훈련
    model.fit(x_train, y_train)

    # 4. 평가, 예측
    
    
    result = model.score(x_train, y_train)

    from sklearn.metrics import accuracy_score
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)

    print(model, " result : ", result)
    print(model, "accuracy-score : ", acc)


banbok(pc)
banbok(lsvc)
banbok(svc)
banbok(knc)
banbok(lr)
banbok(dtc)
banbok(rfc)

#voting으로 해보기




# Perceptron()  result :  0.6126760563380281
# Perceptron() accuracy-score :  0.6388888888888888

# LinearSVC()  result :  0.9295774647887324
# LinearSVC() accuracy-score :  0.9444444444444444

# SVC()  result :  0.7112676056338029
# SVC() accuracy-score :  0.6944444444444444

# KNeighborsClassifier()  result :  0.8098591549295775
# KNeighborsClassifier() accuracy-score :  0.6944444444444444

# LogisticRegression()  result :  0.971830985915493
# LogisticRegression() accuracy-score :  0.9722222222222222

# DecisionTreeClassifier()  result :  1.0
# DecisionTreeClassifier() accuracy-score :  0.9166666666666666

# RandomForestClassifier()  result :  1.0
# RandomForestClassifier() accuracy-score :  1.0