# 가중치를 세이브한다
from tensorflow.keras.models import Sequential
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
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# 모델에대한 정의나 구성은 남아야한다.
# model.load_weights('./_save/keras25.5_save_weights1.h5')
# 에러 보는법 포함.
# weight값까지 구했다는건 fit을 안해도 된다는 소리니까 3번을 주석처리해도 되겠지?
# 주석한 후 실행했더니 에러가 나온다. - 모델이 구성 안되어있으니까 모델 구성해.
# 주석 풀고 돌렸더니. evaluate에서 오류가 난다. = 위에서 weight는 로드했지만 컴파일을 명시를 안 했어.(#3. 부분 명시)
# 컴파일 넣고 돌리니 결과값이 안 좋아.
# model.save_1는 훈련되지 않은 값. 훈련하기도 전에 들어가는 weight의 랜덤값이 들어가 연산되어 엉망인 숫자가 뜬 것.
# loss : 92244.34375  r2스코어 : -9248.012456
# model.save_weights("./_save/keras25.5_save_weights_1.h5")



# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# val_loss 중 가장 낮은 loss 곧 최적의 weight
# checkpoint한 지점의 weight를 빼주면 된다.
# 콜백에서 쓰는 early stopping을 정의(우리는 어디서 체크포인트를 할거야 라는.)
es = EarlyStopping( monitor='val_loss', patience=10, mode='min') # [] 리스트=두개이상 #, restore_best_weights=True 26.2의 영향 = 지웠음 일단
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,filepath='./_ModelCheckPoint/keras26.1_MCP.hdf5') # val_loss가 낮을 수록 좋다는걸 알기에 자동으로.
                                                                                      # 가장 좋은 지점 하나를 세이브해라.
                                                                                      # 체크포인트 저장하는 시점마다 save하면 되겠다. 체크포인트 지점마다 세이브해서 우리에게 줄테지만 결국 마지막 세이브 지점이 가장 최적화된 지점이기에 그 지점만 있으면 되겠다.
                                                                                      # CheckPoint를 지정한 곳에 filepath+파일이름. 해주기(확장자는 h5나 hdf5나 세이브파일일뿐 별차이없다.)
                                                                                      
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=2, callbacks=[es,mcp])
end = time.time() - start

print("===============================================")
print(hist)
print("===============================================")
print(hist.history)        # key-value값의 dictionary형태
                           # model.fit에서 loss와val_loss값을 반환해준다
print("===============================================")
print(hist.history['loss']) 
print("===============================================")
print(hist.history['val_loss'])
print("===============================================")

# plot
plt.figure(figsize=(9, 5))
plt.plot(hist.history['loss'])
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

print("걸린시간 : ", round(end))

# 모델.fit다음에 저장된 25.5save_weight_2 이 놈은 weight가 저장되어 있다.
# 랜덤값이 아닌 훈련이 된 weight결과가 저장된 것이다.
model.save("./_save/keras26_ModelCheckPoint.h5")
# model.save_weights("")
# model.load_weights("./_save/keras25.5_save_weights_.h5")
# 돌려보니 이번엔 loss 58.1993 r20.303692로 정상치로 수렴한다.
# 만약 위에서 compile을 빼고 돌리면 안 돌아간다.
# 컴파일 전에 웨이트를 로드한다면??? 이미 weight값이 들어있다면 상관없음.( 그래도 헷갈리지 않게 컴파일 다음에 명시하자! )

# 모델.save로하면 모델+weight니까 편하긴 한데 용량이 조금 크다.

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( x_test )
#print("예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict) #ypredict test비교
print('r2스코어 : ', r2)

# result
# loss :  60.12312316894531
# r2스코어 :  0.272267782102913
