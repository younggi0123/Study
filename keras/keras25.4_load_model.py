# 25.3모델을 로드한다


from tensorflow.keras.models import Sequential,load_model
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
#print(x.shape)  #feature=13
#print(y.shape)  #feature= 1

# Train&test&val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=66)

# 2. 모델구성
# Model Set
model = Sequential()
# Model Add
# model.add(Dense(70, input_dim=13))
# model.add(Dense(55))
# model.add(Dense(40))
# model.add(Dense(25))
# model.add(Dense(10))
# model.add(Dense(1))
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# model=load_model("./_save/keras25.1_save_model.h5")
# model=load_model("./_save/keras25.3_save_model.h5")



# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2, verbose=2)
end = time.time() - start

print("걸린시간 : ", round(end))

# 25.1과는 위치의 차이이다.?>
# 옛날에 게임하면 세이브 파일 생성 되듯이 모델 저장파일을 생성한다.
# h5라는 확장자를 가진 파일로 생성하겠다.
# _save 폴더를 생성하여 저장토록 해준다.

model=load_model("./_save/keras25.3_load_model.h5")
# model=load_model("./_save/keras25.3_save_model.h5")
 
 
# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( x_test )
#print("예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict) #ypredict test비교
print('r2스코어 : ', r2)


# 일단 이름바꾸지만 후에 save된 시간까지 넣을 것이다
# acc=str(loss[1]model.save(f"./_save/keras24_dacon_save_model_{acc}.h5"))
# acc=str(round(loss[1]model.save(f"./_save/keras24_dacon_save_model_{acc}.h5")))
# f쓰고""하면 "{변수}를 출력"
# 응용 acc=str(loss[1]model.save(f"./_save/keras24_dacon_save_model_{acc}.h5"))
# acc에 . 들어가면 파일저장하고 나중에 쓸 때 문제생길 수 있어서
# acc= str(loss[1]).replace(".","_")    