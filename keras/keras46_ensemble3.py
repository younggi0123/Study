#                                                                           [ 앙상블 훈련3 ]

import numpy as np

x1 = np.array( [range(100), range(301, 401)])                           # ex. 삼성 저가, 고가

x1 = np.transpose(x1)

# 1001부터 1101까지를 맞추겠다.
y1 = np.array( range(1001, 1101) )      # ex. 삼성전자 종가
y2 = np.array( range(101, 201) )        # ex. 하이닉스 종가
y3 = np.array( range(401, 501) )        # ex. 로우닉스 종가

############################ 위 데이터를 넣어서 수정하시오 ! ############################

print(x1.shape, y1.shape, y2.shape)  # (100, 2) (100,) (100,)

# 훈련 part
from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x1, y1, y2, y3, train_size=0.7, random_state=42)
print(x1_train.shape, x1_test.shape)      # (70, 2) (30, 2)
print(y1_train.shape, y1_test.shape)      # (70, ) (30, )
print(y2_train.shape, y2_test.shape)      # (70, ) (30, )
print(y3_train.shape, y3_test.shape)      # (70, ) (30, )

# 2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# 함수형 모델은 layer에 대한 이름을 명시해 줘야 한다.
# 2-1. 모델 no.1
input1 = Input( shape=(2,) )
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3)

# # 2-2. 모델 no.2
# input2 = Input( shape=(3,) )
# dense11 = Dense(10, activation='relu', name='dense11')(input2)
# dense12 = Dense(10, activation='relu', name='dense12')(dense11)
# dense13 = Dense(10, activation='relu', name='dense13')(dense12)
# dense14 = Dense(10, activation='relu', name='dense14')(dense13)
# output2 = Dense(5, activation='relu', name='output2')(dense14)


from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate( [output1] )

# 2-3. output모델1
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

# 2-3. output모델2
output31 = Dense(7)(merge1)
output32 = Dense(11)(output31)
output33 = Dense(11)(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)

# 2-3. output모델3
output41 = Dense(7)(merge1)
output42 = Dense(11)(output41)
output43 = Dense(11)(output42)
output44 = Dense(11, activation='relu')(output43)
last_output3 = Dense(1)(output44)



# 함수형모델은 모델의 정의를 젤끝에서, 시퀀스는 젤 앞에서 한다.
model = Model(inputs=[input1], outputs=[ last_output1, last_output2, last_output3 ])


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])    # metrics에 mae를 넣으니 loss가 다섯개가 나올것.

import time
start = time.time()
model.fit( x1_train, [y1_train, y2_train, y3_train], epochs=10, batch_size=1, verbose=1, validation_split=0.3 )
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')


# 4. 평가, 예측
# Evaluate
loss = model.evaluate( x1_test, [y1_test, y2_test, y3_test])
print('loss : ', loss)

result = model.predict( x1_test )
result = np.array(result).reshape(3,30)
# print(result)


# np array shape를 3으로 맞춰줘야 r2를 쓸 수 있겠지?
from sklearn.metrics import r2_score
r2 = r2_score([y1_test, y2_test, y3_test],result)
print('r2스코어 : ', r2)

# y1_r2 = r2_score( y1_test, result[0] ) #ypredict test비교
# y2_r2 = r2_score( y2_test, result[1] ) #ypredict test비교
# y3_r2 = r2_score( y3_test, result[2] ) #ypredict test비교
# print('y1_r2스코어 : ', y1_r2)
# print('y2_r2스코어 : ', y2_r2)
# print('y3_r2스코어 : ', y3_r2)

# shape이 안 맞으니까 각 각 나온걸 짤라서 하나씩해서 세개뽑아서 한묶음따리하면 되겠지?
# result값 안의 인덱스를 지정해서 출력 혹은? 그냥 한줄ressult로 넣고
# 혹은 위에 result = np.array(x1_test).reshape(3,30)로 리쉐잎해서 ㄱㄱ


# concatenate 소문자 & 대문자 차이
# summary 찍어보고 이해
# 마지막까지 완성해서 결과값.

# 걸린시간 :  54.217 초
# loss :  0.00013953683082945645
# r2스코어 :  0.9999998283529659

