# hdevstudy.tistory.com/157
# https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ko
'''
Keras 모델은 다중 구성 요소로 이루어집니다.

모델에 포함된 레이어 및 레이어의 연결 방법을 지정하는 아키텍처 또는 구성
가중치 값의 집합("모델의 상태")
옵티마이저(모델을 컴파일하여 정의)
모델을 컴파일링하거나 add_loss() 또는 add_metric()을 호출하여 정의된 손실 및 메트릭의 집합
Keras API를 사용하면 이러한 조각을 한 번에 디스크에 저장하거나 선택적으로 일부만 저장할 수 있습니다.

TensorFlow SavedModel 형식(또는 이전 Keras H5 형식)으로 모든 것을 단일 아카이브에 저장합니다. 이것이 표준 관행입니다.
일반적으로 JSON 파일로 아키텍처 및 구성만 저장합니다.
가중치 값만 저장합니다. 이것은 일반적으로 모델을 훈련할 때 사용됩니다.
언제 사용해야 하는지, 어떻게 동작하는 것인지 각각 살펴봅시다.


※ Keras 모델 저장하기
model = ...  # Get model (Sequential, Functional Model, or Model subclass) model.save('path/to/location')

※ 모델을 다시 로딩하기
from tensorflow import keras model = keras.models.load_model('path/to/location')


'''

# 모델을 세이브한다


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


# 옛날에 게임하면 세이브 파일 생성 되듯이 모델 저장파일을 생성한다.
# h5라는 확장자를 가진 파일로 생성하겠다.
# _save 폴더를 생성하여 저장토록 해준다.
model.save("./_save/keras25.1_save_model.h5")
 


'''

# 3. 컴파일, 훈련
# Compile
model.compile(loss='mse', optimizer='adam')
start = time.time()
# History <- Fit
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2, verbose=2)
end = time.time() - start

print("걸린시간 : ", round(end))

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# Predict
y_predict = model.predict( x_test )
#print("예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict) #ypredict test비교
print('r2스코어 : ', r2)
'''