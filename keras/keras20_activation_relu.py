# 신경망 layer계층에서 : linear 다 사용 가능(default임) / sigmoid 마지막도 중간층도 다 가능 / softmax 마지막에만 가능

# 【 relu 】
# relu : 성능 good!
# activation이 주로 하는 일 : layer의 결과값을 다음레이어로 전달할 때 값을 한정지어줌 y=wx+b로 연산된 값이 3개로 전달된 값이
# 그냥 linear였음 값자체가 폭등,폭락 가능 위로 쭉쭉갈수 있으니까. 마지막에 영향받는게 예를들어 이진분류이면 마지막엔 결국
# 예를들어 100만 1000만 2000만 계속 커졌었더라도 결국 마지막에도 1에 수렴할 것이다. 중간에 시그모이드를 넣어주면 중간에 값을 안정시키며 0과 1로 수렴될 수 있다
# 값이 다음으로 전달할때 양수면 그냥두고 -면 산개되는 문제가 생겼다 그래서 나가는 값을 음수를 다 빼버려 봤더니 히든레이어 안에서 성능이 좋아지더라.
# 음수는 다 빼버리고 양수로만 연산을 해보자!
# 값이 다음레이어로 전달될때 양수로만 한정한다. 음수일 땐 0주고 양수일땐 그대로 쓴다
# 다음 노드로 넘어가며 통과한 값은 음수는 0으로, 양수는 그대로. 전달. 이것이 'relu'라는 activation이다.
# 실제로 다음노드로 전달될 때 y=wx+b로 전달될 것이다. 이때 y'=relu(wx+b)라는 식으로 relu로 rapping해준다.
# Q) 음수를 제거하는게 왜 좋나요? 몰라. 성능이 좋다니깐 당분간은 그냥 쓴다.

# relu넣었을때 좋아지는지 판단해본다. 넣다고 무조건 좋아지는 것도 아니니 테스트 해본다.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터 
x = np.array([1.2,3])
y = np.array([1.2,3])


# 2. 모델구성
# 히든레이어의 크기는 어떻게 정하나여?(https://www.clien.net/service/board/kin/10588915)
#(https://data-newbie.tistory.com/140)
model = Sequential()
model.add(Dense(5, input_dim=1)) #default activation = 'lenear'
model.add(Dense(21))
model.add(Dense(9, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(21, activation='relu'))   # 'sigmoid'는 중간에서 사용 가능함
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='linear'))
model.add(Dense(11, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

#########################################################################


# 3. 컴파일,  훈련
# epoch 30 fix
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=32)



# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)


'''
# relu_1차_
loss :  0.0395972765982151
4의 예측값 :  [[3.5832663]]

# relu_2차_epoch100, batch10 조정
loss :  0.02962881326675415
4의 예측값 :  [[3.7452679]]

# relu_3차_epoch100, batch1 조정
loss :  0.008892785757780075
4의 예측값 :  [[3.8694685]]

# relu_4차
loss :  0.007419907487928867
4의 예측값 :  [[3.8742945]]

# relu_5차
loss :  0.005383366718888283
4의 예측값 :  [[3.8913045]]

# relu_6차_레이어 변경
model.add(Dense(21))
model.add(Dense(9, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(21, activation='relu'))   # 'sigmoid'는 중간에서 사용 가능함
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
loss :  0.004118898883461952
4의 예측값 :  [[3.9006293]


# relu_7차
loss :  5.395318112277891e-07
4의 예측값 :  [[3.9988024]]
#################### 7차에서 약 3.9988의 유의미한 결과를 보인다. ####################





#
Dense 5가들어가는 노드에 sigmoid 활성함수를 넣어보면 성능이 급하락한다
model.add(Dense(5, activation='sigmoid'))
loss :  0.13550400733947754
4의 예측값 :  [[2.7151697]]

똑같은 자리에 linear넣어보니
loss :  0.002182356547564268
4의 예측값 :  [[3.9251645]]
성능이 다시 향상되었다.

'''