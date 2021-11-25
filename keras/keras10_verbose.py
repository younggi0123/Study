# 강의 내용: verbose의 쓰임 !

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.9, shuffle=True, random_state=1) 
#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=1))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(25))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(34))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time() # 이지점의 시간을 반환해서.start란 변수에 저장한다.
model.fit(x,y, epochs=1000, batch_size=1, verbose=1)
end = time.time() - start
print("걸린시간 : ", end)



#verbose=0일때 아얘 보이지 않고, 1일때 전체 다 보임, 2일때 프로그래스 바빼고 숫자만 보임
#3일때 epoch만 나온다 #4일때 epoch만 #일때 epoch만...(3이후로는 epoch만 나온다.)
#verbose를 통해 작업 실행 화면을 송출하지 않게 된다 #보여주는게 default임
#verbose로 가려 놓았을 때도 돌아가는지 확인하는 방법은 작업관리자-성능-GPU탭에서 전용GPU메모리 사용량 + 
#Video Decode를 눌러서 CUDA로 바꿔본다 
#데이터의 수가 많아진다면 속도지연이 발생할 수 있다
#BECAUSE 터미널에서 사람이 눈으로 볼수있게 송출하려면 잠깐의 딜레이가 필요하다(컴퓨터의 속도가 너무 빨라 동체시력이 못 따라갈 수 있으니까)


"""
    verbose
    0: 없다
    1: 다
    2: loss까지
    3: epoch만
    
걸린시간 :  23.98872137069702


"""


'''
#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score   #( mean_squred)
r2 = r2_score(y, y_predict) #ypredict test비교
print('r2스코어 : ', r2)

'''