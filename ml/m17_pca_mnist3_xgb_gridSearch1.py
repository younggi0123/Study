# standard 스케일러가 더 잘먹히는 부분이 있을 수 있다.



# n_component >0.95 이상
# xgboost, gridSearch 또는 RandomSearch를 쓸 것

# m16 mnist 결과를 뛰어넘을 것 


# parameters = [
#         {'n_estimators' : [100, 200, 300], 'max_depth' : [4, 5, 6], 'learning_Rate' : [0.1, 0.3, 0.001, 0.01] },
#         {'n_estimators' : [90, 100, 110], 'max_depth' : [4, 5, 6], 'learning_Rate' : [0.1, 0.3, 0.001, 0.01], 'colsample_bytree' : [0.6,0.9,1] },
#         {'n_estimators' : [90, 110], 'max_depth' : [4, 5, 6], 'learning_Rate' : [0.1, 0.001, 0.5], 'colsample_bytree' : [0.6, 0.9, 1], 'colsample_bylevel':[0.6, 0.7, 0.9] },
# ]
# n_jobs = -1

# 실습 go!




# 실습
# 아까 4가지로 모델 만들기
# 784개 DNN으로 만든 것(최상의 성능인 것 // 0.978 이상)과 비교!

# time 체크 (fit에서 잴 것)

# < 예상 결과 >

# 나의 최고의 DNN
# time = ????
# acc = ????

# 2. 나의 최고의 CNN 
# time = ????
# acc = ????

# 3. PCA 0.95
# time = ????
# acc = ????

# 4. PCA 0.99
# time = ????
# acc = ????

# 5. PCA 0.999
# time = ????
# acc = ????

# 4. PCA 1.0
# time = ????
# acc = ????


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import validation
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

# 1.2. 스케일링
# scaler = MinMaxScaler()   
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)     
x_test = scaler.transform(x_test)

# # 1.1.append로 행으로 붙임
# x = np.append( x_train, x_test, axis=0 )
# x = np.reshape(x, (x.shape[0], (x.shape[1]*x.shape[2])) )

# 1.2. PCA
n_components = 154
pca = PCA(n_components)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
# print(cumsum)
# print(np.argmax(cumsum >= 0.95)+1 )
# print(np.argmax(cumsum >= 0.99)+1 )
# print(np.argmax(cumsum >= 0.999)+1 )
# print(np.argmax(cumsum) +1)         #154

# parameters = {"XGB__n_estimators":[90,100,110,200,300], "XGB__learning_rate":[0.1,0.3,0.001,0.01],"XGB__max_depth":[4,5,6],'XGB__use_label_encoder':[False],
#             "XGB__colsample_bytree":[0.6,0.9,1],"XGB__colsample_bylevel":[0.6,0.7,0.9],"XGB__random_state":[66],"XGB__eval_metric":['error']}

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
# model.fit(x_train, y_train, eval_metric='merror')
model.fit(x_train, y_train)

end = time.time()

# 4. 평가, 예측
# result = model.score(x_train, y_train)

# from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)

# print("걸린시간 : ", end - start)
# print("model.score : ", result)
# print("accuracy_score : ", acc)

# # RandomizedSearchCV, StandardScaler, XGBClassifier
# # 걸린시간 :  3900.2147431373596
# # model.score :  1.0
# # accuracy_score :  0.9619

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)
print("최적의 매개변수 : ", model.best_estimator_)   
print("최적의 파라미터 : ", model.best_params_)      
print("best_score_ : ", model.best_score_)            
print("model.score : ", model.score(x_test, y_test))  
y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))   
y_pred_best = model.best_estimator_.predict(x_test)    
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))
print("time : ", end - start)
# print(model.feature_importances_)






# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score
# model besT_score