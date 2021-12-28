# 앙상블 훈련


import numpy as np

x1 = np.array( [range(100), range(301, 401)])                           # ex. 삼성 저가, 고가
x2 = np.array( [range(101, 201),range(411, 511), range(100, 200)] )     # ex. 미국선물 시가, 고가, 종가

# 모델링 수월하게 할 수 있도록.
# 100행 2열로
x1 = np.transpose(x1)

# 100행 3열로
x2 = np.transpose(x2)

# 1001부터 1101까지를 맞추겠다.
y = np.array( range(1001, 1101) )     # ex. 삼성전자 종가

# shape이 헷갈리거나 틀릴 수 있으니 찍어본다.
print(x1.shape, x2.shape, y.shape)  # (100, 2) (100,3) (100, )

# 훈련 part
from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, random_state=42)
print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)    # (70, 3) (30, 3)
print(y_train.shape, y_test.shape)      # (70, ) (30, )

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

# 2-1. 모델 no.2
input2 = Input( shape=(3,) )
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(5, activation='relu', name='output2')(dense14)

# concatenate = 잇다(like 사슬)  # 대소문자 차이 = 함수, 클래스
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate( [output1, output2] )
merge2 = Dense(10, activation='relu')(merge1)
# merge3 = Dense(10, activation='relu')(merge2)
# merge4 = Dense(10, activation='relu')(merge3)
merge5 = Dense(7)(merge2)
last_output = Dense(1)(merge5)                  #linear => 1

# 함수형모델은 모델의 정의를 젤끝에서, 시퀀스는 젤 앞에서 한다.
model = Model(inputs=[input1, input2], outputs= last_output)

########################################[ S U M M A R Y ]########################################
# model.summary()
'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 2)]          0
__________________________________________________________________________________________________
dense11 (Dense)                 (None, 10)           40          input_2[0][0]
__________________________________________________________________________________________________
dense1 (Dense)                  (None, 5)            15          input_1[0][0]
__________________________________________________________________________________________________
dense12 (Dense)                 (None, 10)           110         dense11[0][0]
__________________________________________________________________________________________________
dense2 (Dense)                  (None, 7)            42          dense1[0][0]
__________________________________________________________________________________________________
dense13 (Dense)                 (None, 10)           110         dense12[0][0]
__________________________________________________________________________________________________
dense3 (Dense)                  (None, 7)            56          dense2[0][0]
__________________________________________________________________________________________________
dense14 (Dense)                 (None, 10)           110         dense13[0][0]
__________________________________________________________________________________________________
output1 (Dense)                 (None, 7)            56          dense3[0][0]
__________________________________________________________________________________________________
output2 (Dense)                 (None, 5)            55          dense14[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 12)           0           output1[0][0]
                                                                 output2[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 10)           130         concatenate[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 7)            77          dense[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            8           dense_1[0][0]
==================================================================================================
Total params: 809
Trainable params: 809
Non-trainable params: 0
___________________________________________________
'''

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])     # metrics로 'mae'를 넣어 본다.
import time
start = time.time()
model.fit( [x1_train, x2_train], y_train, epochs=50, batch_size=1, verbose=1, validation_split=0.3 )
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')


# 4. 평가, 예측
# Evaluate
loss = model.evaluate( [x1_test, x2_test], y_test)
print('loss : ', loss)

result = model.predict( [x1_test, x2_test] )
print("비교용 : ")
print(result)


from sklearn.metrics import r2_score
r2 = r2_score( y_test, result ) #ypredict test비교
print('r2스코어 : ', r2)

# 컴파일할때 metrics로 mae넣어보면 로스값이 5개가 나올텐데 뭐냐면,
# 전체loss(loss들의 합임), dens3loss,  dense8loss,  dense3의mae, dense8의mae





# concatenate 소문자 & 대문자 차이
# summary 찍어보고 이해
# 마지막까지 완성해서 결과값.

# 걸린시간 :  54.217 초
# loss :  0.00013953683082945645
# r2스코어 :  0.9999998283529659

