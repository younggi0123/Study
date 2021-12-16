# - Bidirectional -
# LSTM의 단방향적 문제를 시간지남에 따라 데이터가 소실되는 부분이 생기지만
# 반대편으로 양방향으로 진행시 가중치가 더해지기에 성능이 좋을 것이다.란 이론
# 어떤 RNN을 사용할지 명시해줘야 함. => wrapping
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional, Conv1D, Flatten
from tensorflow.python.keras.backend import flatten

# 1. 데이터
x = np.array(
            [[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5],
             [4, 5, 6]]
            )
y= np.array( [4, 5, 6, 7] )

print(x.shape, y.shape) # (4, 3) (4, )

x = x.reshape(4, 3, 1)

# 2. 모델 구성
# bias전 (2*2)*input 까지 conv연산. bias가 output만큼 연산된다는 것
# kernel size만큼 곱해주겠다
# params = ((2*2)*input+bias) x output
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(3,1)))
model.add(Dense(10))
model.add(Flatten())    # Flatten위치 바뀜
model.add(Dense(1))

# 결과는 2차원이여야 한다.(결과치 확인)

model.summary()


# CNN의 연속형 조각된 커널처럼 RNN도 부분부분 짤라서 오른쪽으로 가는 식으로 가능하다
# 통상적으로 유사하거나 LSTM tune을 별로로 하면 CNN1D가 나을 때도 있다
# 머신2개면 하나씩 돌릴 수도 있고 시간과 자원을 생각해서 돌려야 한다.(시간이 반 이하로 걸리니 촉박한 경우, 얘로 두번 돌려야 할 수도 있으니까)
# SHAPE만 맞으면 CNN을 못 쓸때 RNN으로 쓸 수도 있는 법이다. INPUT : CONV2D=4차원, CONV1D=3차원, RNN=3차원
# 차원의 입력이 RNN과 CONV1D는 같으므로 RESHAPE 없이 쓸수있지만 OUTPUT이 2차원인 RNN과 달리 CONV은 OUTPUT이 3차원이기에 DENSE연결시 FLATTEN해줘야 한다.
# 허나, ★☆FLATTEN을 하지 않는 방법은 RNN으로 엮는 것이다 (섞어쓰기 가능).★☆


#                                           ※   DNN / RNN / CNN 정리   ※
#
#                                                             【◀───────────────Input───────────────▶Ⅱ◀─Output─▶】
# ┌─────────┬─────────┬──────────────┬───────────┬─────────────┬────────┬────────┬─────────┬────────────┬────────────┐
# │   ＼    │Data구조 │ input_shape  │  output   │ dense와 연결│  4차원 │  3차원 │  2차원  │    1차원   │   아웃풋    │
# ├─────────┼─────────┼──────────────┼───────────┼─────────────┼────────┼────────┼─────────┼────────────┼────────────┤
# │ Dense   │   2dim  │      1dim    │   2dim    │             │        │        │  batch  │ input_dim  │   units    │
# ├─────────┼─────────┼──────────────┼───────────┼─────────────┼────────┼────────┼─────────┼────────────┼────────────┤
# │ LSTM    │   3dim  │      2dim    │   2dim    │             │        │  batch │timesteps│  feature   │   filter   │
# ├─────────┼─────────┼──────────────┼───────────┼─────────────┼────────┼────────┼─────────┼────────────┼────────────┤
# │ Conv1D  │   3dim  │      2dim    │   3dim    │ Flatten/RNN │        │  batch │   steps │  input_dim │   filter   │
# ├─────────┼─────────┼──────────────┼───────────┼─────────────┼────────┼────────┼─────────┼────────────┼────────────┤
# │ Conv2D  │   4dim  │      3dim    │   4dim    │   Flatten   │ batch  │   row  │  column │   chanel   │   filter   │
# └─────────┴─────────┴──────────────┴───────────┴─────────────┴────────┴────────┴─────────┴────────────┴────────────┘
# timesteps/steps는 출력차이(4차 3차)만 있는 것. flatten을 하거나 펴줘야하는 부분만 다름

















'''
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

# 4. 평가, 예측
model.evaluate(x, y)
result = model.predict( [[[5], [6], [7]]] )
print(result)

'''