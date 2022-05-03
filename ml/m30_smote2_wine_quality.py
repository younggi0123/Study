# 그냥 증폭해서 성능비교 
# 총 7개 라벨을 증폭.
# x_train을 증폭하고 x_test는 철저히 평가해석만 하여 성능비교

# ※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★ 데이터 증폭 수업 ※☆★※☆★※☆★※☆★※☆★※☆★※☆★※☆★

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score


# 1. 데이터
# 데이터 불러오기
path = "../_data/kaggle/wine/"
# read_csv takes a sep param, in your case just pass sep=';' like so:
# data = read_csv(csv_path, sep=';')
datasets = pd.read_csv(path+"winequality-white.csv", index_col=None, header=0, sep=';')#첫째줄이 헤더고 헤더가 있음
# index_col's default is 'None'

# 판다스
y = datasets['quality']
x = datasets.drop(['quality'], axis =1)
print(x.shape, y.shape) #(4898, 11) (4898,)

print(pd.Series(y).value_counts())      # y는 하나니까 series잖아(dataframe아니잖아) # 판다스
# 6    2198
# 5    1457
# 7     880
# 8     175
# 4     163
# 3      20
# 9       5
# Name: quality, dtype: int64

# 넘파이화
datasets = datasets.values
x = datasets[: , :11]
y = datasets[: , 11]
print(x.shape, y.shape)

print(y)
# [6. 6. 6. ... 6. 7. 6.]

x_train, x_test, y_train, y_test \
    = train_test_split(x, y, train_size=0.75,
    shuffle=True, random_state=66, stratify=y )

#####################################################################################
#                                    ※ 기본 ※
#####################################################################################

# model = XGBClassifier(n_jobs=4)
# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)
# print("model.score : ",
#       round(score,4) )
# y_predict = model.predict(x_test)
# print("accuracy score : ",
#       round(accuracy_score(y_test, y_predict), 4 ) )
# print('f1_score : ',
#       round(f1_score(y_test, y_predict, average='macro'), 4) )
# # print('f1_score : ',f1_score(y_test, y_predict, average='micro'))

# # accuracy score :  0.6433
# # f1_score :  0.3853





#####################################################################################
#                                 ※ 데이터 증폭 ※
#####################################################################################

#                                    S M O T E

# print("==================================SMOTE적용==================================")
# 에러발생
# https://stackoverflow.com/questions/45943335/smote-value-error

# I had a similiar issue.
# SMOTE is based in a KNN algorithm, so you need a minimal number of samples to create a new instance of this subset.
# For example:
# If you is trying to predict is a integer value, class 1, 2, 3, and supposing that you have just 2 samples of class 1,
# how to get k-3 neighbors? Will be impossible. It's too umbalanced!!
# The message is pretty clear:
# Expected n_neighbors <= n_samples.
# So, you need have more or equals SAMPLES than neighbors, to create new instances.
# I look yout dataset and you have just 4 samples of OUTPUT 1. So, the message is saying you have just 4 but I need 6 neighbors to create a new instance of them.


# k-neighbor의 주위값 카피 알고리즘을 따르겠지?
# 그런데 이러한 에러가 발생한다.  ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
# 하나의 데이터가 있을 때 근접이웃값으로 채워줌
# n_samples가 4인데 k_neighbors = 를 그거보다 작게하라는..
# k_neighbors=3 추가함

# 근대 앞에서 했던 column shrink하면 이거 안 해도 또 돌아간다능..

# 그래도 test는 건들면 안 되는거 알지?
smote = SMOTE(random_state=66, k_neighbors=2)


x_train, y_train = smote.fit_resample(x_train, y_train) # fit resample에 xtrain ytrain 넣은 것을 반환해줘야
print( pd.Series(y_train).value_counts() )
# 1    53
# 0    44
# 2    14

# 다시 53개에 맞추고자 한다.


model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ",
      round(score,4) )
y_predict = model.predict(x_test)
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ) )
print('f1_score : ',
      round(f1_score(y_test, y_predict, average='macro'), 4) )
# print('f1_score : ',f1_score(y_test, y_predict, average='micro'))


# SMOTE 후    <- 데이터 증폭
# accuracy score :  0.6433
# f1_score :  0.3844
