from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List ,Dict, Tuple


import sklearn 
from sklearn.model_selection import StratifiedKFold , KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score 
import pandas as pd
from catboost import Pool,CatBoostClassifier
import warnings
 
warnings.filterwarnings(action='ignore')

path = "../_data/dacon/jobcare_data/"
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
sample = pd.read_csv(path + 'sample_submission.csv')


code_d = pd.read_csv(path + '속성_D_코드.csv')
code_h = pd.read_csv(path + '속성_H_코드.csv')#.iloc[:,:-1]
code_l = pd.read_csv(path + '속성_L_코드.csv')

SEED = 42


print(code_d.shape)                 # (1114, 4)
print(train.shape, test.shape)      # (501951, 35) (46404, 34)

# code_d.columns= ["속성 D 코드","속성 D 세분류코드","속성 D 소분류코드","속성 D 중분류코드","속성 D 대분류코드"]
# code_h.columns= ["속성 H 코드","속성 H 중분류코드"]

code_d.columns= ["attribute_d","attribute_d_d","attribute_d_s","attribute_d_m", "attribute_d_l"]
code_h.columns= ["attribute_h","attribute_h_p","attribute_h_l"]
code_l.columns= ["attribute_l","attribute_l_d","attribute_l_s","attribute_l_m","attribute_l_l"]
print(code_d.head())

def merge_codes(df:pd.DataFrame,df_code:pd.DataFrame,col:str)->pd.DataFrame:
    df = df.copy()
    df_code = df_code.copy()
    df_code = df_code.add_prefix(f"{col}_")
    df_code.columns.values[0] = col
    return pd.merge(df,df_code,how="left",on=col)

def preprocess_data(
                    df:pd.DataFrame,is_train:bool = True, cols_merge:List[Tuple[str,pd.DataFrame]] = []  , cols_equi:List[Tuple[str,str]]= [] ,
                    cols_drop:List[str] = ["id","person_prefer_f","person_prefer_g" ,"contents_open_dt"]
                    ) -> Tuple[pd.DataFrame,np.ndarray]:
    df = df.copy()

    y_data = None
    if is_train:
        y_data = df["target"].to_numpy()
        df = df.drop(columns="target")

    for col, df_code in cols_merge:
        df = merge_codes(df,df_code,col)

    cols = df.select_dtypes(bool).columns.tolist()
    df[cols] = df[cols].astype(int)

    for col1, col2 in cols_equi:
        df[f"{col1}_{col2}"] = (df[col1] == df[col2] ).astype(int)

    df = df.drop(columns=cols_drop)
    return (df , y_data)


# 소분류 중분류 대분류 속성코드 merge 컬럼명 및 데이터 프레임 리스트
cols_merge = [
              ("person_prefer_d_1" , code_d),
              ("person_prefer_d_2" , code_d),
              ("person_prefer_d_3" , code_d),
              ("contents_attribute_d" , code_d),
              ("person_prefer_h_1" , code_h),
              ("person_prefer_h_2" , code_h),
              ("person_prefer_h_3" , code_h),
              ("contents_attribute_h" , code_h),
              ("contents_attribute_l" , code_l),
]

# 회원 속성과 콘텐츠 속성의 동일한 코드 여부에 대한 컬럼명 리스트
cols_equi = [

    ("contents_attribute_c","person_prefer_c"),
    ("contents_attribute_e","person_prefer_e"),

    ("person_prefer_d_2_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_2_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_2_attribute_d_l" , "contents_attribute_d_attribute_d_l"),
    ("person_prefer_d_3_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_3_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_3_attribute_d_l" , "contents_attribute_d_attribute_d_l"),

    ("person_prefer_h_1_attribute_h_p" , "contents_attribute_h_attribute_h_p"),
    ("person_prefer_h_2_attribute_h_p" , "contents_attribute_h_attribute_h_p"),
    ("person_prefer_h_3_attribute_h_p" , "contents_attribute_h_attribute_h_p"),

]

# 학습에 필요없는 컬럼 리스트
cols_drop = ["id","person_prefer_f","person_prefer_g" ,"contents_open_dt", "contents_rn", "person_rn"]

x_train, y_train = preprocess_data(train, cols_merge = cols_merge , cols_equi= cols_equi , cols_drop = cols_drop)
x_test, _ = preprocess_data(test,is_train = False, cols_merge = cols_merge , cols_equi= cols_equi  , cols_drop = cols_drop)
print(x_train.shape , y_train.shape , x_test.shape)  # (501951, 68) (501951,) (46404, 68) <-이거 아님
                                                     # (501951, 67) (501951,) (46404, 67)

import time, os
start = time.time()
# SMOTE화 하기
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)
# 증폭된 y 컬럼 라벨별 행개수 파악
print(np.unique(y_train, return_counts=True))
print(x_train.shape , y_train.shape , x_test.shape)  #
print("SMOTE 걸린시간 : ", time.time()-start)








cat_features = x_train.columns[x_train.nunique() > 2].tolist()

is_holdout = False
n_splits = 5
# iterations = 6000
patience = 400

cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

scores = []
models = []

# 캣부스트 핵심
# 구글검색:What’s so special about CatBoost?
# https://velog.io/@tyhlife/Whats-so-special-about-CatBoost

# https://hanishrohit.medium.com/whats-so-special-about-catboost-335d64d754ae
# 하이퍼 파라미터 튜닝
# https://catboost.ai/en/docs/concepts/parameter-tuning
# https://data-newbie.tistory.com/159
# https://dailyheumsi.tistory.com/136
random_state= SEED
task_type="GPU"
# devices = 0 
use_best_model = True
eval_metric ="F1"
cat_features = cat_features
parameters1 = [{
                'iterations' : [ 3000, 5000 ],
                # 'iterations' : [3000, 5000],
                'leaf_estimation_iterations' : [10,20, 30],
                'max_depth' : [6,7,8],
                'random_strength' : [0, 50, 100],
                'learning_rate' : [0.02, 0.05, 0.1, 0.15, 0.2],
                # 'one_hot_max_size' : [3,4,5,6,7],
                # 'random_state' : [42],
                'task_type' : [task_type],
                # 'devices' : [devices],
                'use_best_model' : [use_best_model],
                'eval_metric' : [eval_metric],
                'cat_features' : [cat_features]
              }]




# https://stackoverflow.com/questions/60648547/how-to-increase-accuracy-of-model-using-catboost
parameters2 = [{
            'depth':[3,1,2,6,4,5,7],
            'iterations':[3000],
              
            #   'iterations':[3000,5000],
              'learning_rate':[0.03,0.01,0.1,0.2,0.3],
              'l2_leaf_reg':[3,1,5,10,100],
            # #   'border_count':[32,5,10,20,50,100,200],
            # #   'bagging_temperature':[0.03,0.09,0.25,0.75],
            #   'random_strength':[0.2,0.5,0.8],
            # #   'max_ctr_complexity':[1,2,3,4,5]
              }]


# 수정 참고 https://stackoverflow.com/questions/60648547/how-to-increase-accuracy-of-model-using-catboost
# grid_search_result = model.grid_search(params,
#     X=train_set, y=train_label, cv=5, partition_random_seed=3, stratified=True)
# classifier = CatBoostClassifier()
# model = classifier.grid_search(parameters1,cv=cv,verbose=2,refit=True)

import os, time
start = time.time()
models = []
for tri, vai in cv.split(x_train):
    print("="*50)
    preds = []
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★https://www.inflearn.com/questions/155560
    # refit이 True면 bestIndex를 기반으로 return하기때문에 gridSearcvCV에서 get best 파라미터가 안 먹힌다
    # model =GridSearchCV( CatBoostClassifier(), parameters1, cv=cv , verbose=2, refit=True)
    # model =GridSearchCV( CatBoostClassifier(), parameters2, cv=cv , verbose=2, refit=False)
    
    # model =GridSearchCV( CatBoostClassifier(), parameters1, cv=cv , verbose=2, refit=True)
    model =GridSearchCV( CatBoostClassifier(), parameters2, cv=cv , verbose=2, refit=True)


    # model = CatBoostClassifier()
    model.fit(x_train.iloc[tri], y_train[tri], 
            eval_set=[(x_train.iloc[vai], y_train[vai])], 
            early_stopping_rounds=patience ,
            verbose = 100
        )
    models.append(model)
    # scores.append(model.get_best_score()["validation"]["F1"])
    # scores.append(model.best_score_["validation"]["F1"])
    
    if is_holdout:
        break

end = time.time() - start
print("걸린시간 : ", round(end, 2), '초')
print(scores)
print(np.mean(scores))

# ==================================================
# Learning rate set to 0.027144
# 0:      learn: 0.6395404        test: 0.6415329 best: 0.6415329 (0)     total: 391ms    remaining: 19m 34s
# 100:    learn: 0.6437034        test: 0.6518881 best: 0.6518881 (100)   total: 12.7s    remaining: 6m 3s
# 200:    learn: 0.6548709        test: 0.6710498 best: 0.6712356 (199)   total: 24.9s    remaining: 5m 46s
# 300:    learn: 0.6608502        test: 0.6791284 best: 0.6792267 (298)   total: 36.8s    remaining: 5m 29s
# 400:    learn: 0.6646407        test: 0.6817098 best: 0.6819598 (378)   total: 48.5s    remaining: 5m 14s
# 500:    learn: 0.6674827        test: 0.6824116 best: 0.6827287 (466)   total: 59.9s    remaining: 4m 58s
# 600:    learn: 0.6698834        test: 0.6830426 best: 0.6831905 (554)   total: 1m 11s   remaining: 4m 45s
# bestTest = 0.6831905499
# bestIteration = 554
# Shrink model to first 555 iterations.
# ==================================================
# Learning rate set to 0.027144
# 0:      learn: 0.6202637        test: 0.6188115 best: 0.6188115 (0)     total: 378ms    remaining: 18m 53s
# 100:    learn: 0.6435191        test: 0.6504979 best: 0.6505994 (99)    total: 12.8s    remaining: 6m 7s
# 200:    learn: 0.6563656        test: 0.6703403 best: 0.6703403 (200)   total: 25.3s    remaining: 5m 51s
# 300:    learn: 0.6614119        test: 0.6782425 best: 0.6783629 (288)   total: 37.2s    remaining: 5m 33s
# 400:    learn: 0.6648736        test: 0.6807426 best: 0.6808487 (397)   total: 49s      remaining: 5m 17s
# 500:    learn: 0.6674464        test: 0.6814387 best: 0.6819125 (467)   total: 1m       remaining: 5m 1s
# bestTest = 0.681912525
# bestIteration = 467
# Shrink model to first 468 iterations.
# ==================================================
# Learning rate set to 0.027144
# 0:      learn: 0.6202532        test: 0.6201275 best: 0.6201275 (0)     total: 427ms    remaining: 21m 21s
# 100:    learn: 0.6420214        test: 0.6504379 best: 0.6504379 (100)   total: 13.2s    remaining: 6m 18s
# 200:    learn: 0.6541332        test: 0.6716300 best: 0.6716300 (200)   total: 25.8s    remaining: 5m 59s
# 300:    learn: 0.6608362        test: 0.6802124 best: 0.6802124 (300)   total: 37.8s    remaining: 5m 39s
# 400:    learn: 0.6645453        test: 0.6829935 best: 0.6829935 (400)   total: 49.7s    remaining: 5m 22s
# bestTest = 0.6833033573
# bestIteration = 410
# Shrink model to first 411 iterations.
# ==================================================
# Learning rate set to 0.027144
# 0:      learn: 0.6080430        test: 0.6130987 best: 0.6130987 (0)     total: 392ms    remaining: 19m 35s
# 100:    learn: 0.6435771        test: 0.6532148 best: 0.6532148 (100)   total: 13s      remaining: 6m 12s
# 200:    learn: 0.6558559        test: 0.6743917 best: 0.6743917 (200)   total: 25.5s    remaining: 5m 54s
# 300:    learn: 0.6617948        test: 0.6824021 best: 0.6824021 (300)   total: 37.5s    remaining: 5m 35s
# 400:    learn: 0.6651601        test: 0.6839316 best: 0.6841236 (394)   total: 49.2s    remaining: 5m 18s
# 500:    learn: 0.6677177        test: 0.6840329 best: 0.6846585 (463)   total: 1m       remaining: 5m 3s
# bestTest = 0.6846585285
# bestIteration = 463
# Shrink model to first 464 iterations.
# ==================================================
# Learning rate set to 0.027144
# 0:      learn: 0.6185799        test: 0.6197662 best: 0.6197662 (0)     total: 340ms    remaining: 16m 59s
# 100:    learn: 0.6433598        test: 0.6512024 best: 0.6512024 (100)   total: 13s      remaining: 6m 14s
# 200:    learn: 0.6546255        test: 0.6703188 best: 0.6703987 (198)   total: 25.5s    remaining: 5m 55s
# 300:    learn: 0.6611251        test: 0.6800639 best: 0.6801919 (298)   total: 37.5s    remaining: 5m 36s
# 400:    learn: 0.6649987        test: 0.6827033 best: 0.6828618 (392)   total: 49.2s    remaining: 5m 19s
# 500:    learn: 0.6675963        test: 0.6838425 best: 0.6839156 (496)   total: 1m 1s    remaining: 5m 4s
# 600:    learn: 0.6705345        test: 0.6843531 best: 0.6846070 (589)   total: 1m 12s   remaining: 4m 50s
# bestTest = 0.6846070435
# bestIteration = 589
# Shrink model to first 590 iterations.
# [0.6831905498735705, 0.6819125250009433, 0.6833033573141487, 0.6846585285384362, 0.6846070435151572]
# 0.6835344008484512

threshold = 0.35

pred_list = []
scores = []
for i,(tri, vai) in enumerate( cv.split(x_train) ):
    pred = models[i].predict_proba(x_train.iloc[vai])[:, 1]
    pred = np.where(pred >= threshold , 1, 0)
    score = f1_score(y_train[vai],pred)
    scores.append(score)
    pred = models[i].predict_proba(x_test)[:, 1]
    pred_list.append(pred)
print(scores)
print(np.mean(scores))

pred = np.mean( pred_list , axis = 0 )
pred = np.where(pred >= threshold , 1, 0)

sample_submission = pd.read_csv(f'{path}sample_submission.csv')
sample_submission['target'] = pred
print(sample_submission)
sample_submission.to_csv('catboost_220125.csv', index=False)
