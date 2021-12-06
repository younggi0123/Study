# 가중치를 세이브한다
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.utils import validation
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import time
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
# Data load
datasets = load_boston()
# Train_set
x = datasets.data
y = datasets.target
# print(x.shape)  #feature=13
# print(y.shape)  #feature= 1

# Train&test&val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=66)

# 2. 모델구성
# Model Set
model = Sequential()

# 3. 컴파일, 훈련
# Compile
# cp에 모델과 weight가 들어있으면 로드모델했을때 다 포함되어 있으니 컴파일도 필요없을 것이다
model.compile(loss='mse', optimizer='adam') 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( monitor='val_loss', patience=10, mode='min', restore_best_weights=False ) # [] 리스트=두개이상
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,filepath='./_ModelCheckPoint/keras26.3_MCP.hdf5') # val_loss가 낮을 수록 좋다는걸 알기에 자동으로.
                                                                                      # 가장 좋은 지점 하나를 세이브해라.
                                                                                      # 체크포인트 저장하는 시점마다 save하면 되겠다. 체크포인트 지점마다 세이브해서 우리에게 줄테지만 결국 마지막 세이브 지점이 가장 최적화된 지점이기에 그 지점만 있으면 되겠다.
                                                                                      # CheckPoint를 지정한 곳에 filepath+파일이름. 해주기(확장자는 h5나 hdf5나 세이브파일일뿐 별차이없다.)
                                                                                      
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=2, callbacks=[es,mcp])
end = time.time() - start

print("===============================================")
print(hist.history['val_loss'])
print("===============================================")

print("걸린시간 : ", round(end))

model.save("./_save/keras26.1_save_model.h5")

#  세이브버전
model = load_model("./_save/keras26_ModelCheckPoint.h5")
#  체크포인트 버전
# model = load_model('./_ModelCheckPoint/keras26.1_MCP.hdf5')#복습할때 파일명 일치시키기!!!!!!!!!!

# 4. 평가, 예측
print("======================== 1. 기본 출력 ========================")
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# Predict
y_predict = model.predict( x_test )
r2 = r2_score(y_test, y_predict) #y-predict test비교
print('r2스코어 : ', r2)

print("======================== 2. load_model 출력 ========================")
model2 = load_model('./_save/keras26.3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss : ', loss2)

# Predict
y_predict2 = model2.predict( x_test )
r2 = r2_score(y_test, y_predict2) #y-predict test비교
print('r2스코어 : ', r2)


print("======================== 3. ModelCheckPoint 출력 ========================")
model3 = load_model('./_ModelCheckPoint/keras26.3_MCP.h5')
loss3 = model3.evaluate(x_test, y_test)
print('loss : ', loss3)

# Predict
y_predict3 = model2.predict( x_test )
r2 = r2_score(y_test, y_predict3) #y-predict test비교
print('r2스코어 : ', r2)