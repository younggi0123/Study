import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 데이터
a = np.array( range(1, 101) )
x_predict = np.array( range(96, 106) )

size = 5    #  x 4개 y 1개

# y가 predict 값이겠죠?

############################################# 함수 사용 부분 #############################################
def split_x(dataset, size):                     # 함수 선언(알고리즘)
    aaa = []                                    # 빈 리스트 선언
    for i in range( len(dataset) - size + 1):   # length함수(dataset만큼 - 자를길이 + 1)    => dataset까지 만큼 for문 돌면 역전되어서 [9:5] 같이 되어버리니까 안 되겠지?
        subset = dataset[i : (i+size)]          # 부분집합( 1줄 ) = dataset 중 1부터 5까지 채워넣겠다.
        aaa.append(subset)                      # 빈 리스트 aaa에 subset을 붙이겠다.
    return np.array(aaa)                        # aaa값을 리턴해서 다음 for문을 진행. 다시 2~6, 3~7 ㄱㄱ

bbb= split_x(a, size)
# print(bbb)
# print(bbb.shape)          # (96, 5)    6행 5열
x= bbb[:, :4]              # ~4열까지
y= bbb[:, 4]               # 인덱스 4열
# print(x, y)
# print(x.shape,y.shape)    # (96, 4)(96, )
#  [:, : ] #        :하나면 모든행.     :두개면 모든행+모든열
# [a:b, c:d] 앞에건 행a:b까지, 뒤에건 열 c:d까지
# [a, b] 인덱스 a지정, b지정
# [:,  d]  인덱스 d열지정
##########################################################################################################


pred= split_x(x_predict, 5)   # 함수 사용하기

x_pred = pred[:, :4]           # 0123   4  ㅣ인덱스 지정
# print(x_pred)                   # 5번째 열인 100 101 102 103 104 105 가 출력됨
x_pred = x_pred.reshape(6,4,1) #(6,1)
print(x_pred.shape)

x = x.reshape( x.shape[0], 4, 1 )
print(x)
# 2. 모델 구성
model = Sequential()
model.add(LSTM(32, return_sequences=False, activation='tanh', input_shape=(3, 1)) ) #LSTM's DEFAULT = tanh!!!!!!!!!!!!!!!
model.add(Dense(48, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()

model.fit(x, y, epochs=100)

end = time.time() - start
print("걸린시간 : ", round(end),"sec")


# 4. 평가, 예측
model.evaluate(x, y)
result = model.predict(x_pred)
print(result)


'''
[[ 98.31231 ]
 [ 98.927475]
 [ 99.50668 ]
 [100.059654]
 [100.58682 ]
 [101.088684]]
 '''