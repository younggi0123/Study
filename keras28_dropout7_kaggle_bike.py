# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】
# Sequential vs model

# 하고 캐글에 제출
# bike_data는 로그변환해야 되는거 알지? 근데 얘는 로그변환하면 rmse가 너무 확 떨어지긴 해

# File 7. Bike_Sharing_Demand Data

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# RSME 정의
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


# 1. 데이터
# Data load
path = "./_data/titanic/"
path = "./_data/bike/"
train = pd.read_csv(path + 'train.csv')
# print(train)        # (10866, 12)
test_file = pd.read_csv(path + 'test.csv')          # test라고 지으면 애매해서
# print(test_file)         # (6493, 9)      train에서 casual,regis,count의 3개가 빠진 데이터
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

# x & y 설정
x = train.drop( ['datetime', 'casual', 'registered', 'count'], axis=1) # 이렇게 4개 빼고 컬럼 구성
test_file = test_file.drop( ['datetime'], axis=1) # model.predict에서 돌아가게 하도록 datetime 오브젝트를 지운다.
y = train['count']

# 로그변환
y = np.log1p(y)

# Train& Test& Val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.7)#, shuffle=True, random_state=66

# Preprocessing
# 전처리 4대장
# a. MinMaxScaler
# scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
# d. MaxAbsScaler
scaler = MaxAbsScaler()
# # Scaler fit & transform
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_test_file = scaler.transform(test_file) # ★실수 잦은 부분★ test.csv도 스케일링해야됨.bike데이터는 평소train안에서만 하는거랑 다르게 따로 파일로 하는거니까.
                                          # +)  모 든  x 는  스 켈 링 되 어 야  한 다 ! ! ! ! !

# 2. 모델링 구성
# Sequential Ver Set
model = Sequential()
model.add(Dense(12, input_dim=8))
model.add(Dense(11, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='linear'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))










# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
################################################################################################
# 이부분 이해 안되면 일단 
import datetime
date = datetime.datetime.now()
hello = date.strftime("%m%d_%H%M")      #월일_시분 #1206_0456(날짜_시간)
# print(hello)

filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 2500-0.3724.hdf5 (하단 hist에서 반환한 값이 그대로 저장된다. hist아닌 hist.history)
                                             # 4자리까지 빼겠다
model_path = "".join( [ filepath, 'keras28.7_', hello, '_',filename ] )    # join함수 -> 시간, 파일경로, 파일이름
# ★./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf5
################################################################################################

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( monitor='val_loss', patience=10, mode='min' ) # [] 리스트=두개이상 #, restore_best_weights=True
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path) # 파일 경로명에 model_path로 바꿔준다(위에있는거 끌어와)
import time                                                                                      
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=2, callbacks=[es,mcp])
end = time.time() - start

print("걸린시간 : ", round(end))

model.save('./_Save/keras28.7_save_bike.h5')




# 4. 평가, 예측
print("======================= 1. 기본출력 =======================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test)

# Predict
r2 = r2_score(y_test, y_pred)
print('r2스코어 : ', r2)

# RMSLE와 비슷한 효과의 rmse이다.
rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)

print("====================== 2. load_model 출력 =======================")
model2 = load_model('./_Save/keras28.7_save_bike.h5')
loss2 = model2.evaluate(x_test, y_test)
print("loss : ", loss2)
y_pred2 = model.predict(x_test)

# Predict
r2 = r2_score(y_test, y_pred2)
print('r2스코어 : ', r2)

# RMSLE와 비슷한 효과의 rmse이다.
rmse = RMSE(y_test, y_pred2)


print("====================== 3. ModelCheckPoint 출력 =======================")
model3 = load_model(model_path)
loss3 = model3.evaluate(x_test, y_test)
print("loss : ", loss3)
y_pred3 = model.predict(x_test)

# Predict
r2 = r2_score(y_test, y_pred3)
print('r2스코어 : ', r2)

# RMSLE와 비슷한 효과의 rmse이다.
rmse = RMSE(y_test, y_pred3)





############################################제출파일 생성############################################
# results = model.predict(x_test_file)       # x_test_file이 들어감에 유의
# submit_file['count'] = results           # test파일에서 예측한걸 count로 나오면 submit파일에 들어가진다

# print(submit_file[:10])                 # submit_file이라는 변수에 count가 들어갔고 csv로 보내준다.

# submit_file.to_csv(path + "bike_preprocessingTest_submit_ver.csv", index=False)



#드랍아웃 적용시 미적용시

# 1. 드랍아웃 미적용 : loss :  36.39024353027344 r2 스코어 :  0.5595313606671943

# 2. 드랍아웃 적용 :  55.588287353515625 r2 스코어 :  0.3271576401020574


