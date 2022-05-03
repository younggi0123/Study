
# 【       Subject : 22'. 02. 07. ReduceLR을 적용+ GlobalAveragePooling 써보기       】
# 외부 - White Wine Data sets 기반(+ keras wine datasets)


# File 5. Wine Data

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장 소환 !
import numpy as np
import pandas as pd


# 1. 데이터
# Data load


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


y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, random_state=66, stratify=y)                        # stratify 넣어야겠중?

# a. MinMaxScaler
# scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
# d. MaxAbsScaler
# scaler = MaxAbsScaler()
# # Scaler fit & transform
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)           # train은 minmaxscaler가 됨
# x_test = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                    # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)

# 2. 모델링

# Sequential Ver Set
model = Sequential()
model.add(Dense(160, input_dim=11))
model.add(Dense(140, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='softmax'))               # ★ activation은 softmax ★ activation은 softmax ★ activation은 softmax ★





from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(lr=learning_rate)







# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)


# MODEL CHECK POINT
################################################################################################
# import datetime
# date = datetime.datetime.now()
# hello = date.strftime("%m%d_%H%M")      #월일_시분 #1206_0456(날짜_시간)
# # print(hello)

# filepath = "./_ModelCheckPoint/"
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 2500-0.3724.hdf5 (하단 hist에서 반환한 값이 그대로 저장된다. hist아닌 hist.history)
#                                              # 4자리까지 빼겠다
# model_path = "".join( [ filepath, 'keras28.5_', hello, '_',filename ] )    # join함수 -> 시간, 파일경로, 파일이름
# ★./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf5
################################################################################################


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( monitor='val_loss', patience=10, mode='min' ) # [] 리스트=두개이상 #, restore_best_weights=True
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path) # 파일 경로명에 model_path로 바꿔준다(위에있는거 끌어와)


from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)





import time                                                                     
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=2, callbacks=[es,reduce_lr])
end = time.time() - start


print("걸린시간 : ", round(end))

model.save('./_Save/keras28.5_save_wine.h5')

# 4. 평가, 예측
# print("======================= 1. 기본출력 =======================")
# Evaluate
loss = model.evaluate(x_test, y_test)




print("#########################################################################################################")
print('lerning rate : ', learning_rate)
print('loss : ', round(loss[0],4))
print('accuracy : ', round(loss[1], 4))
print("걸린시간 : ", round(end))



# 기존 :  55.588287353515625 r2 스코어 :  0.3271576401020574



#########################################################################################################
# 적용 후
# lerning rate :  0.01
# loss :  1.2955
# accuracy :  0.449
# 걸린시간 :  20