# 【       Subject : 21'. 12. 01. keras23의 함수형 모델을 기존의 Sequential 대신 적용해본다.       】

# Sequential vs model

# File 6. Fetch_covtype Data

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np



# 1. 데이터
# Data load
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(np.unique(y))               # 1 2 3 4 5 7 , 7개의 라벨
y = to_categorical(y)

# 최소값 최대값
# print(np.min(x), np.max(x))     #  -173.0 7173.0

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, random_state=66)

# print(x.shape)                    # (581012, 54)
# print(y.shape)                    # (581012, 8)
# print(x_train.shape, y_train.shape) #(464809, 54) (464809, 8)
# print(x_test.shape, y_test.shape)   #(116203, 54) (116203, 8)

# API 땡겨올때는 어떤 전처리를 사용할 것인지 정의를 해줘야한다.
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
x_train = scaler.transform(x_train)           # train은 minmaxscaler가 됨
x_test = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                    # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)

# 2. 모델링
# Sequential Ver Set
model = Sequential()
model.add(Dense(70, input_dim=54))         #input_dim 54될 것
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(8, activation='softmax'))               # ★ activation은 softmax ★ activation은 softmax ★ activation은 softmax ★




# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)

################################################################################################
# 이부분 이해 안되면 일단 
import datetime
date = datetime.datetime.now()
hello = date.strftime("%m%d_%H%M")      #월일_시분 #1206_0456(날짜_시간)
# print(hello)

filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 2500-0.3724.hdf5 (하단 hist에서 반환한 값이 그대로 저장된다. hist아닌 hist.history)
                                             # 4자리까지 빼겠다
model_path = "".join( [ filepath, 'keras28.6_', hello, '_',filename ] )    # join함수 -> 시간, 파일경로, 파일이름
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

model.save('./_Save/keras28.6_save_fetch_covtype.h5')

# 4. 평가, 예측
# print("======================= 1. 기본출력 =======================")
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★


# print("====================== 2. load_model 출력 =======================")
model2 = load_model('./_Save/keras28.6_save_fetch_covtype.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss : ', loss2[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss2[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★



print("====================== 3. ModelCheckPoint 출력 =======================")
model3 = load_model(model_path)
loss3 = model3.evaluate(x_test, y_test)
print('loss : ', loss3[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss3[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★


#드랍아웃 적용시 미적용시

# 1. 드랍아웃 미적용 : loss :  36.39024353027344 r2 스코어 :  0.5595313606671943

# 2. 드랍아웃 적용 :  55.588287353515625 r2 스코어 :  0.3271576401020574


