# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】

# Sequential vs model

# ※ 전처리 및 소스에 대한 설명은 File 1. 참고 ※


# File 3. Breast Cancer Data

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np

# 1. 데이터
# Data load
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x))     #  0.0 4254.0

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66          )

# print(x.shape)                  # (569, 30)
# print(y.shape)                  # (569, )
# print(y_test[:11])              # 값찍방법 1. 값을 찍어보니 [1 1 1 1 1 0 0 1 1 1 0]로 이진분류이다. 시그모이드 ㄱㄱ
# print(np.unique(y))             # 값찍방법 2. [0 1]

# API 땡겨올때는 어떤 전처리를 사용할 것인지 정의를 해줘야한다.
# 전처리 4대장
# a. MinMaxScaler
scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
# d. MaxAbsScaler
# scaler = MaxAbsScaler()
# Scaler fit & transform
scaler.fit(x_train)
x_train = scaler.transform(x_train)           # train은 minmaxscaler가 됨
x_test = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                    # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)

# 2. 모델링

# Sequential Ver Set
model = Sequential()
model.add(Dense(20, input_dim=30))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))               # ★ 마지막은 반드시 시그모이드 ★ 마지막은 반드시 시그모이드 ★ 마지막은 반드시 시그모이드 ★


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )               # 이진분류 인지 - binary_crossentropy

################################################################################################
# 이부분 이해 안되면 일단 
import datetime
date = datetime.datetime.now()
hello = date.strftime("%m%d_%H%M")      #월일_시분 #1206_0456(날짜_시간)
# print(hello)

filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 2500-0.3724.hdf5 (하단 hist에서 반환한 값이 그대로 저장된다. hist아닌 hist.history)
                                             # 4자리까지 빼겠다
model_path = "".join( [ filepath, 'keras27.3_', hello, '_',filename ] )    # join함수 -> 시간, 파일경로, 파일이름
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

model.save('./_Save/keras27.3_save_cancer.h5')

# # 4. 평가, 예측
# print("======================= 1. 기본출력 =======================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# print("====================== 2. load_model 출력 =======================")
# model2 = load_model('./_Save/keras27.3_save_cancer.h5')
# loss2 = model2.evaluate(x_test, y_test)
# print("loss : ", loss2)

print("====================== 3. ModelCheckPoint 출력 =======================")
model3 = load_model(model_path)
loss3 = model3.evaluate(x_test, y_test)
print("loss : ", loss3)



# 1. 기본출력 loss :  다시...

# 2. 모델세이브 loss :  다시...

# 3. 모델체크포인트 loss : 오류 해결 중
