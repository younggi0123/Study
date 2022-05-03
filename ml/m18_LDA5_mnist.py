# LDA로 mnist를 하여 PCA와 비교해본다.

# 무조건 default 아닌 n_components도 설정해봐.


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import validation

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)



# 1.2. 스케일링
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train, y_train)
x_train = scaler.transform(x_train)# 윗 문장이랑 합치면 fit_transform이겠죠?
x_test = scaler.transform(x_test)#test는 fit_transform하면안된다









# # 1.1.append로 행으로 붙임
# x = np.append( x_train, x_test, axis=0 )
# x = np.reshape(x, (x.shape[0], (x.shape[1]*x.shape[2])) )

# 1.2. LDA
n_components = 154
lda = LinearDiscriminantAnalysis()
# lda = LinearDiscriminantAnalysis(n_components)
x_train = lda.fit_transform(x_train,y_train)
x_test = lda.transform(x_test)

# lda_EVR = lda.explained_variance_ratio_
# # print(pca_EVR)
# # print(sum(pca_EVR))

# cumsum = np.cumsum(lda_EVR)
# # print(cumsum)
# # print(np.argmax(cumsum >= 0.95)+1 )
# # print(np.argmax(cumsum >= 0.99)+1 )
# # print(np.argmax(cumsum >= 0.999)+1 )
# # print(np.argmax(cumsum) +1)         #154

hyperparameters = [
        {'xgb__n_estimators' : [100, 200, 300], 'xgb__max_depth' : [4, 5, 6], 'xgb__learning_rate' : [0.1, 0.3, 0.001, 0.01],"xgb__eval_metric":['merror'] },
        {'xgb__n_estimators' : [90, 100, 110], 'xgb__max_depth' : [4, 5, 6], 'xgb__learning_rate' : [0.1, 0.001, 0.01], 'xgb__colsample_bytree' : [0.6, 0.9, 1], "xgb__eval_metric":['merror']},
        {'xgb__n_estimators' : [90, 110], 'xgb__max_depth' : [4, 5, 6], 'xgb__learning_rate' : [0.1, 0.001, 0.5], 'xgb__colsample_bytree' : [0.6, 0.9, 1], 'xgb__colsample_bylevel':[0.6, 0.7, 0.9], }
]

pipe = Pipeline(  [("ss", StandardScaler()), ("xgb", XGBClassifier(use_label_encoder=False))] ) # Pipeline사용법 : 1. 리스트씌우기 2.이름을 명시

# model = GridSearchCV(keras_model, hyperparameters, cv = 1)
# model = RandomizedSearchCV(XGBClassifier(use_label_encoder=False), hyperparameters, cv=3, verbose=1,
#                      refit=True, n_jobs=-1)    # model : 어떤 모델도 다 쓸 수 있다, parameters : 해당 모델의 파라미터 사용

model = RandomizedSearchCV(pipe, hyperparameters, cv=3, verbose=2, refit=True, n_jobs=-1)



#3. 훈련
import os, time
start = time.time()

# eval_metric은 parameter에 추가하는게 좋을 것 같네요! 만약 pipeline으로 엮었을때 fit에 eval_metric을 추가시 warning이 생깁니다.
# model.fit(x_train, y_train, eval_metric='error')
# model.fit(x_train, y_train, eval_metric='merror')
model.fit(x_train, y_train)

end = time.time()

# 4. 평가, 예측
result = model.score(x_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("걸린시간 : ", end - start)
print("model.score : ", result)
print("accuracy_score : ", acc)

# RandomizedSearchCV, StandardScaler, XGBClassifier
# 걸린시간 :  3900.2147431373596
# model.score :  1.0
# accuracy_score :  0.9619