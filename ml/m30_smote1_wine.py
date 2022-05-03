# 데이터 강제 축소 후 다시 증폭하는 파일

# ※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★ 데이터 증폭 수업 ※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
# print(y.value_unique())               # 넘파이

print(pd.Series(y).value_counts())      # y는 하나니까 series잖아(dataframe아니잖아) # 판다스
# 1    71
# 0    59
# 2    48

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
# So, You know It needs to be shuffled
'''

#####################################################################################
#                                    ※ 기본 ※
#####################################################################################

x_train, x_test, y_train, y_test \
    = train_test_split(x, y, train_size=0.75,
    shuffle=True, random_state=66, stratify=y )

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ", score)
y_predict = model.predict(x_test)
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ))
'''
# accuracy score :  0.9778    <- 기본데이터(보통 방법 fit)


#####################################################################################
#                                 ※ 데이터 축소 ※
#####################################################################################


# 데이터셋 중 뒤의 30개를 빼고자 한다
# np.where도 되지만
# slicing도 되잖아!

# x도 해줘야지?
x_new = x[ :-30]
y_new = y[ :-30]
print( pd.Series(y_new).value_counts() )
# 1    71
# 0    59
# 2    18

print(y_new)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]


# 데이터 다시 변경하여!

x_train, x_test, y_train, y_test \
    = train_test_split(x_new, y_new, train_size=0.75,
    shuffle=True, random_state=66, stratify=y_new )
'''
model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ", score)
y_predict = model.predict(x_test)
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ))
'''
# accuracy score :  0.9459    <- 데이터 축소


# 그리하여, parameter만 건드리는 것 보다 데이터 정제가 더 중요하다는걸 알 수 있다 !



# 실전에서 만약에 이런 데이터를 받았다면??? 을 가정으로 축소한 실습 데이터  =>
# 축소된 데이터
# 1    71
# 0    59
# 2    18
# 1라벨의 71을 기준으로 다른 라벨들을 변경하고자 한다.

#####################################################################################
#                                 ※ 데이터 증폭 ※
#####################################################################################

#                                    S M O T E


print("==================================SMOTE적용==================================")

# 그래도 test는 건들면 안 되는거 알지?
smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train) # fit resample에 xtrain ytrain 넣은 것을 반환해줘야
print( pd.Series(y_train).value_counts() )
# 1    53
# 0    44
# 2    14

# 다시 53개에 맞추고자 한다.


model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ", score)
y_predict = model.predict(x_test)
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ))

# accuracy score :  0.973     <- 데이터 증폭



# 데이터가 너무 방대할 경우?? => 데이터를 잘라서 증폭하고 다시 합치는 등의 방법