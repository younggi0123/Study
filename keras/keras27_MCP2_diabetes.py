# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】

# Sequential vs model

# ※ 전처리 및 소스에 대한 설명은 File 1. 참고 ※


# File 2. Diabetes Data

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import validation
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np

# 1. 데이터
# Data load
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x))     #  -0.137767225690012 0.198787989657293

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=49          )

# print(x.shape)                  #(442, 10)
# print(y.shape)                  #(442, )

# API 땡겨올때는 어떤 전처리를 사용할 것인지 정의를 해줘야한다.
# 전처리 4대장
# a. MinMaxScaler
# scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
# d. MaxAbsScaler
# scaler = MaxAbsScaler()
# Scaler fit & transform
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test  = scaler.transform(x_test)

# 2. 모델링 비교
# Sequential Ver Set
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(15, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))




#########################################################################################################
########################################## 여기서 부터 ModelCheckPoint 부분 ##############################
# 3. 컴파일,훈련
# cp에 모델과 weight가 들어있으면 로드모델했을때 다 포함되어 있으니 컴파일도 필요없을 것이다
model.compile(loss='mse', optimizer='adam') 

################################################################################################
# 이부분 이해 안되면 일단 
import datetime
date = datetime.datetime.now()
hello = date.strftime("%m%d_%H%M")      #월일_시분 #1206_0456(날짜_시간)
# print(hello)

filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 2500-0.3724.hdf5 (하단 hist에서 반환한 값이 그대로 저장된다. hist아닌 hist.history)
                                             # 4자리까지 빼겠다
model_path = "".join( [ filepath, 'keras27.2_', hello, '_',filename ] )    # join함수 -> 시간, 파일경로, 파일이름
# ★./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf5
################################################################################################

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( monitor='val_loss', patience=10, mode='min' ) # [] 리스트=두개이상 #, restore_best_weights=True
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path) # 파일 경로명에 model_path로 바꿔준다(위에있는거 끌어와)
                                                                                      
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=2, callbacks=[es,mcp])
end = time.time() - start


print("걸린시간 : ", round(end))

model.save('./_Save/keras27.2_save_diabetes.h5')

# # 4. 평가, 예측
# print("======================= 1. 기본출력 =======================")
# loss = model.evaluate(x_test, y_test)
# print("loss : ", loss)
# result = model.predict(x_test)
# r2 = r2_score(y_test, result)
# print("r2 스코어 : ", r2)

# print("====================== 2. load_model 출력 =======================")
# model2 = load_model('./_Save/keras27.2_save_diabetes.h5')
# loss2 = model2.evaluate(x_test, y_test)
# print("loss : ", loss2)

# result2 = model2.predict(x_test)
# r2 = r2_score(y_test, result2)
# print("r2 스코어 : ", r2)

print("====================== 3. ModelCheckPoint 출력 =======================")
model3 = load_model(model_path)
loss3 = model3.evaluate(x_test, y_test)
print("loss : ", loss3)

result3 = model3.predict(x_test)
r2 = r2_score(y_test, result3)
print("r2 스코어 : ", r2)




# 1. 기본출력 loss :  28301.80859375  r2스코어 :  -4.31158994271296


# 2. 모델세이브 loss :  28301.80859375  r2스코어 :  -4.31158994271296

# 3. 모델체크포인트 loss : 오류 해결 중











