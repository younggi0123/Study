# 【       Subject : 22'. 02. 07. ReduceLR을 적용+ GlobalAveragePooling 써보기       】
# dropout- cancer 기반

# ※ 전처리 및 소스에 대한 설명은 File 1. 참고 ※


# File 3. Breast Cancer Data

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
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
# scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
# d. MaxAbsScaler
# scaler = MaxAbsScaler()
# Scaler fit & transform
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)           # train은 minmaxscaler가 됨
# x_test = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                    # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)

# 2. 모델링

# Sequential Ver Set
model = Sequential()
model.add(Dense(20, input_dim=30))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='linear'))

# model.add(Dropout(0.2))

model.add(Dense(10, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))               # ★ 마지막은 반드시 시그모이드 ★ 마지막은 반드시 시그모이드 ★ 마지막은 반드시 시그모이드 ★

from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(lr=learning_rate)



# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'] )               # 이진분류 인지 - binary_crossentropy



################################################################################################
# # 이부분 이해 안되면 일단 
# import datetime
# date = datetime.datetime.now()
# hello = date.strftime("%m%d_%H%M")      #월일_시분 #1206_0456(날짜_시간)
# # print(hello)

# filepath = "./_ModelCheckPoint/"
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 2500-0.3724.hdf5 (하단 hist에서 반환한 값이 그대로 저장된다. hist아닌 hist.history)
#                                              # 4자리까지 빼겠다
# model_path = "".join( [ filepath, 'keras28.3_', hello, '_',filename ] )    # join함수 -> 시간, 파일경로, 파일이름
# # ★./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf5
# ################################################################################################

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping( monitor='val_loss', patience=10, mode='min' ) # [] 리스트=두개이상 #, restore_best_weights=True
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path) # 파일 경로명에 model_path로 바꿔준다(위에있는거 끌어와)



from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)




import time                                                                                      
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=2, callbacks=[es, reduce_lr]) # mcp
end = time.time() - start
# model.save('./_Save/keras28.3_save_cancer.h5')



loss = model.evaluate(x_test, y_test)


print("#########################################################################################################")
print('lerning rate : ', learning_rate)
print('loss : ', round(loss[0],4))
print('accuracy : ', round(loss[1], 4))
print("걸린시간 : ", round(end))






# 기존 :  loss :  [0.24344804883003235, 0.9064327478408813]     # loss, acc

#########################################################################################################
# lerning rate :  0.01
# loss :  0.2417
# accuracy :  0.9064
# 걸린시간 :  10