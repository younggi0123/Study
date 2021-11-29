# 강의 : KERAS 모델 구조 이해
# mlp : multi layer perceptron

# 오브젝트 위에 화살표 대고 설명 읽어보기
import numpy as np
from tensorflow.keras.models import Sequential
#Sequential 모델은 레이어를 선형으로 연결하여 구성합니다. 레이어 인스턴스를 생성자에게 넘겨줌으로써 Sequential 모델을 구성할 수 있습니다.
from tensorflow.keras.layers import Dense
#시퀀스 오브젝트 model에 노드를 Dense레이어를 통해 연결해줍니다
#Dense 레이어는 입력과 출력을 모두 연결해주며 입력과 출력을 각각 연결해주는 가중치를 포함하고 있습니다.

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,
               1.6,1.5,1.4,1.3]])

y = np.array([11,12,13,14,15,16,17,18,19,20])


# NUMPY 전치행렬(행렬의 행과 열 바꾸기)의 3단계
# (★ 참고 : https://rfriend.tistory.com/289)
# 1. a.T attribute(reshape)
# 2. np.transpose(a) method
# 3. np.swapaxes(a, 0, 1) method

#x1 = np.arange(20).reshape(10, 2)   #1.
#x1 = np.reshape(2, 10)               #1.
x2 = np.transpose(x)                 #2.
x3 = np.swapaxes(x, 0, 1)            #3.

# 찾을때 print(n.shape) 를 자주 쓰며 확인하게 된다. (쉐잎이 맞지 않으면 어차피 안 되니까)

# 원본
print(x.shape)
print(y.shape)
# 변환본
#print(x1.shape)
print(x2.shape)
print(x3.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(3))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x2, y, epochs=500, batch_size=1) #batch_size 열개수?!


#4. 평가, 예측
loss = model.evaluate(x2,y)
print('loss : ', loss)
y_predict = model.predict([[10, 1.3]])
print('[10, 1.3]의 예측값 : ', y_predict)



# 1차) 예측값 (20)
'''
모델
model.add(Dense(10, input_dim=2))
model.add(Dense(3))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(1))
model.fit(x2, y, epochs=500, batch_size=1)
'''
#loss :  0.016393620520830154
#[10, 1.3]의 예측값 :  [[19.728954]]



# 2차) 예측값 (20)
'''
모델
model.add(Dense(20, input_dim=2))
model.add(Dense(3))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(2))
model.fit(x2, y, epochs=500, batch_size=1)
'''
# loss :  0.0003111028636340052
# [10, 1.3]의 예측값 :  [[19.9992]]
# 2차 결과에서 19.9992로 유의미한 결과를 얻음


#                       강의 결론                       #

# 주어진 [10, 1.3]의 1행2열은 열이 predict와 같아야 한다.
# 컬럼, 디멘션 개수는 같아야 한다
# 특성의 개수 = 열의 개수 = 피처개수
# 1행 2열이면 대괄호 개수는 2개 x의 인풋디멘션 컬럼개수와 y_predict의 사이즈가 같아야
# 행이 백개든 천개든 열의 사이즈만 맞으면 상관 없다.

# ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 
# ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 
# ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 
# ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 
# ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 암기 : 열 우선 행 무시 ★★ 
