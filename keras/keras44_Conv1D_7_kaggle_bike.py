# reshape등 변환할때 숫자 매번 기입하는 것 보다
# x_train[0] , x[0]등 인덱스 지정으로 쓰면 직접 바꾸지 않아도 되니 편함(ㄱㄱ)


# https://wooono.tistory.com/175
# ▲Pandas DataFrame을 numpy 배열로 변환하는 방법
# 하지만 본 데이터에서는, drop으로 문자열 데이터 등 dataframe에서 문제가 될만한 것들을 제거 해줬으니까 수치형 데이터만 남았고,
# pandas에 대한 수치화 데이터는 전부 numpy로 되어있기에 train한 데이터는 당연히 수치데이터니까 numpy겠지?
# 그렇기 때문에 다시 to.numpy할 필요가 없는 것이다.

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM , Conv1D, Flatten

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = read_csv(path+'train.csv')  
print(train.shape)      # (10886, 12)
test_file = read_csv(path+'test.csv')
print(test_file.shape)    # (6493, 9)
submit_file = read_csv(path+ 'sampleSubmission.csv')
print(submit_file.shape)     # (6493, 2)


x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']

# feature가 12였으나 drop으로 4개를 빼서 8이될것임


# print(x)
# print(y)
# 로그변환
y = np.log1p(y)
# print(y)


# print(x.shape, y.shape) # (10886,12) (10886, )

x = x.values
# print(x.shape)
x = x.reshape(10886, 8, 1)      #feature를 맞추어 reshape해준다.
# 

# reshape 시도 시" python - "DataFrame"개체에는 'reshape'특성이 없습니다. " 라는 오류 발생.
# pandas.dataframe에는 내장 reshape 메소드가 없지만 .values를 사용하여 기본 numpy 배열 객체에 액세스하고 reshape를 호출한다.
# 참고 : https://pythonq.com/so/python/554998


# print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size =0.7, shuffle=True, random_state = 42)


#2. 모델구성
model = Sequential() 
model.add(Conv1D(7, 2, input_shape =(8,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Flatten())
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=False)
# mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
#                        filepath = './_ModelCheckPoint/keras27_7_MCP.hdf5')

import time
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.3, callbacks=[es])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')



model.save('./_save/keras27_7_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print ('r2 :', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)

# loss :  36.39024353027344 r2 스코어 :  0.5595313606671943


# Conv1D 수행 시
# 걸린시간 :  6.317 초
# loss :  1.4532485008239746
# r2 : 0.2696466930702849
# RMSE :  1.2055076107730558