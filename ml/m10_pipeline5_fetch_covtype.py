import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler

# Data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)

# 2. 모델구성
from sklearn.ensemble import RandomForestClassifier

# 파이프라인 사용 방법
from sklearn.pipeline import make_pipeline, Pipeline


#2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())

#3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("result : ", result)
# result :  0.9999978485786635
print("accuracy-score : ", acc)
# accuracy-score :  0.9554142319905682