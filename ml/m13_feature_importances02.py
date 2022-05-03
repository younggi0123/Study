# 13_1번을 가져다가
# 첫 번째 컬럼을 제거 후
# 13_1번과 성능 비교 ㄱㄱ

import numpy as np
from sklearn.datasets import load_iris

# Data
datasets = load_iris()
x = datasets.data

x = np.delete(x,3,axis=1) # 0부터 순차적 열 삭제
# numpy delete !
# 리스트로 묶어서 [0,1]하면 두 개도 뺄 수 있슴
# x = np.delete(x,[0,1],axis=1)


y = datasets.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성

# feature는 tree계열에만 있어
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = DecisionTreeClassifier()
# model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print(model, "accuracy-score : ", acc)
print(model.feature_importances_)

# accuracy-score :  0.9666666666666667
# m13_feature_importances1 출력값
# [0.         0.0125026  0.03213177 0.95536562]         # 합이 1이다.



# 자원의 효율성 -> 0인 feature을 뺏을때 나머지 세개로 돌렸을때 0.96이 나오는가?
#                                  ↓
# 첫 번째 컬럼 삭제
# 속도 , 자원 up
# DecisionTreeClassifier() accuracy-score :  0.9666666666666667
# [0.0125026  0.03213177 0.95536562]

# 두번째 컬럼 삭제
# DecisionTreeClassifier() accuracy-score :  0.9333333333333333
# [0.0125026  0.03213177 0.95536562]

# 세번째 컬럼 삭제
# DecisionTreeClassifier() accuracy-score :  0.9
# [0.01599432 0.05393981 0.93006587]

# 네번째 컬럼 삭제
# DecisionTreeClassifier() accuracy-score :  0.9333333333333333
# [0.0678641  0.01458346 0.91755244]