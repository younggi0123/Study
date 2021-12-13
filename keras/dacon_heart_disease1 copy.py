from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from pandas import get_dummies
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers
from lightgbm import LGBMClassifier

#평가 산식 : f1 score
from sklearn import metrics
def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score

#1 데이터

path = "../_data/dacon/heart_disease/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + "sample_submission.csv")

# 데이터 요약통계 확인
# print(train.describe())

# 데이터 구조 확인
# print(train.shape)  # (151, 15)

# 데이터 변수 확인
# print(train.columns)  # Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        # 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
                        # dtype='object')


# ※ 환자 데이터 살펴보기 - 환자 데이터를 시각화하여, 데이터의 특징을 직관적으로 이해

# 151명의 환자에게서 심장병 유무를 확인
# f = sns.countplot(x=train['target'], data=train)
# f.set_title("Heart disease presence distribution")
# f.set_xticklabels( ['No Heart disease', 'Heart Disease'] )
# plt.xlabel("")
# plt.show()

# 151명의 환자에서 심장병 유무를 남녀로 나누어 확인
# f = sns.countplot(x=train['target'], data=train, hue=train['sex'])
# plt.legend(['Female', 'Male'])
# f.set_title("Heart disease presence by gender")
# f.set_xticklabels( ["No Heart Disease", "Heart Disease"] )
# plt.xlabel("")
# plt.show()

# 변수(Feature) 사이에서의 상관관계 정도를 Heatmap으로 구현
# +1에 가까울 수록 => Positive Correlation
# -1에 가까울 수록 => Negative Correlation
# heat_map = sns.heatmap(train.corr(method='pearson'), annot=True, fmt='.2f', linewidths=2)
# heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45)
# plt.show()

# 높은거
# target, thal
# slope,oldpeak
# target,exang
# target,oldpeak,
# target,ca
# target,cp
# cp,exang
# age,thalach
# exang, cp

# 낮은거
# id시리즈
# restecg
# trestbps
# chol     

plt.scatter(train['oldpeak'],train['slope'])
plt.show()

'''
# 데이터 전처리
# 보유한 데이터는 범주형/숫자형의 혼합 형태로, 원활한 텐서플로우 모델링을 위해 데이터 전처리가 필요함
# 머신러닝 모델에서는 범주형/숫자형을 떠나 모든 Feature들이 숫자로 처리 됨

# feature column은 'raw 데이터'와 '모델링하는 데이터'를 연결짓는 브릿지 역할
feature_columns =[]

# 수치형 열(Numeric col)은 실수값을 변형시키지 않고 그대로 전달
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
    feature_columns.append(tf.feature_column.numeric_column(header))
    
# 버킷형 열(Bucketized column)은 수치값을 구간을 나누어 범주형으로 변환
age = tf.feature_column.numeric_column("age")
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# 범주형 열(Categorical column)은 특정 문자열을 수치형으로 매핑하여 전달
train["thal"] = train["thal"].apply(str)
thal = tf.feature_column.categorical_column_with_vocabulary_list("thal",['3','6','7'])
thal_one_hot = tf.feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

train["sex"] = train["sex"].apply(str)
sex = tf.feature_column.categorical_column_with_vocabulary_list("sex",['0','1'])
sex_one_hot = tf.feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)

train["cp"] = train["cp"].apply(str)
cp = tf.feature_column.categorical_column_with_vocabulary_list("cp",['0','1','2','3'])
cp_one_hot = tf.feature_column.indicator_column(cp)
feature_columns.append(cp_one_hot)

train["slope"] = train["slope"].apply(str)
slope = tf.feature_column.categorical_column_with_vocabulary_list("slope",['0','1','2'])
slope_one_hot = tf.feature_column.indicator_column(slope)
feature_columns.append(slope_one_hot)

# 임베딩 열(Embedding column)은 범주형 열에 가능한 값이 많을 때 사용
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# 교차특성 열(Crossed column)은 여러 특성을 연결하여 하나의 특성으로 만듦
age_thal_crossed = tf.feature_column.crossed_column( [age_buckets, thal], hash_bucket_size=1000 )
age_thal_crossed = tf.feature_column.indicator_column(age_thal_crossed)
feature_columns.append(age_thal_crossed)

cp_slope_crossed = tf.feature_column.crossed_column( [cp, slope], hash_bucket_size=1000 )
cp_slope_crossed = tf.feature_column.indicator_column(cp_slope_crossed)
feature_columns.append(cp_slope_crossed)


# Pandas 데이터 프레임 => Tensorflow 데이터셋
def create_dataset(dataframe, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(train['target'])
    return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))\
        .shuffle(buffer_size=len(dataframe))\
        .batch(batch_size)
        
# 전체 데이터를 Training set과 Test set으로 나누기
y = train['target']
x = train.drop(['id','target'], axis=1)
test_file = test_file.drop(['id'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle = True, random_state = 42)
# train_ds = create_dataset(x_train)
# test_ds = create_dataset(x_test)

# print(x_train)
# print(y_train)
# 여기까지 오류 x
# # 2. 모델링
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

# model = tf.keras.models.Sequential([
#   tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
#   tf.keras.layers.Dense(units=128, activation='relu'),
#   tf.keras.layers.Dropout(rate=0.2),
#   tf.keras.layers.Dense(units=128, activation='relu'),
#   tf.keras.layers.Dense(units=1, activation='sigmoid')
# ])


# model.summary()
#여기까지 오류 x 다음부터 오류 o

# 컴파일 fit
# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).
# https://datascience.stackexchange.com/questions/82440/valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupported-object-type

# print(x_train.value_counts())
# print(y_train.value_counts())
# print(x_test.value_counts())
# print(y_test.value_counts())

x_train=np.asarray(x_train).astype(np.int)
y_train=np.asarray(y_train).astype(np.int)
x_test=np.asarray(x_test).astype(np.int)
y_test=np.asarray(y_test).astype(np.int)
# print(x_train.shape)#135,13
# print(y_train.shape)#135

# 컴파일, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )  
history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

# 모델의 정확도 시각화
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.ylim((0, 1))
# plt.legend(['train', 'test'], loc='upper left');
# plt.show()

# 모델의 손실 시각화
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# 심장질환 예측
# 만들어진 DNN 모델로 심장질환을 예측
# 앞서 얻은 accuracy에 상관없이, precision/recall/f1-score가 높지 않음을 확인
# precision = 정밀도, recall = 재현율, f1-score = 정밀도와 재현율의 조화평균
from sklearn.metrics import classification_report, confusion_matrix
# # y_pred = model.predict(x_test)      #밑에 중복
# # bin_pred = tf.round(y_pred).numpy().flatten()   #밑에 중복 #binary pred
# print( classification_report(y_test[:], bin_pred) )  #array형식의 값들이라 어레이로 추출.

# 평가

y_pred = model.predict(x_test)
bin_pred = tf.round(y_pred).numpy().flatten()   #binary pred
# bin_pred = y_pred.round(0).astype(int) #.reshape((31, ))
# print( classification_report(y_test[:], bin_pred) )  


loss = model.evaluate(x_test, y_test)
# result = model.predict(test_file)
result = model.predict(test_file).round(0).astype(int)
f1 = f1_score(bin_pred, y_test)

print("loss : ",loss[0])
print("accuracy : ",loss[1])
print("f1 : ", f1)

# Confusion matrix(혼동 행렬)로 결과 시각화
# class_names = [0,1]
# fig,ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks,class_names)
# plt.yticks(tick_marks,class_names)
# from sklearn.metrics import confusion_matrix

# cnf_matrix = confusion_matrix(y_test, bin_predictions)
# sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="Blues",fmt="d",cbar=False)
# ax.xaxis.set_label_position('top')
# plt.tight_layout()
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# plt.show()

################################ 제출용 ########################################
submission['target'] = result
submission.to_csv(path + "heart_disease1213_2.csv", index = False)



'''