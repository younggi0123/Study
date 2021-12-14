# 과제는 하단에 !

# 통상 RNN은 LSTM임.
# 또 읽엉https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
#        https://ratsgo.github.io/deep%20learning/2017/10/10/RNNsty/
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

# 1. 데이터
x = np.array([[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5],
             [4, 5, 6]]
             )
y= np.array( [4, 5, 6, 7] )
print(x.shape, y.shape) # (4, 3) (4, )

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행        , 열       , 몇 개씩 자르는지!!!)
x = x.reshape(4, 3, 1)


model = Sequential()
model.add(LSTM(10, activation='linear', input_shape=(3, 1)) )  # SimpleRNN만 LSTM으로 바꿔준다.
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()\
# 1. 왜 param이 480인가? 왜 simpleRNN보다 4배나 연산량이 늘어났는가?
# 2. 구글에 LSTM으로 검색시 딱 나오는 https://docs.likejazz.com/lstm/를 이해하고, 이미지에서 activation tahn의 역할이 무엇인가???
# 3. GATE의 개념 정리



# 1.딥러닝의 경우 backpropagation을 통해 학습시키면 기울기 값이 소멸하는 문제가 발생한다. RNN의 경우 멀리 떨어져 있는 요소끼리 큰 연관성을 가질 수 있지만 기울기 소멸 문제로 인해 그러한 영향력을 반영하기 힘들다. Long-term dependency 문제를 해결하기 위해 gated RNN(gate : 입력값에 따라 0과 1 사이의 값을 가지며, layer간 정보 전달의 여부를 결정하는 요소)이 제안되었고, 그 중 대표적인 모델로 LSTM과 GRU가 있다.
# 답: https://www.quora.com/Do-LSTMs-have-significantly-more-parameters-than-standard-RNNs
# ◎ LSTM(Long Short Term Memory)는 3개의 gate(input gate, forget gate, output gate)를 가지는 모델로 일반 RNN보다 4배의 매개변수를 학습시켜야 함으로.
#     →  https://ark-hive.tistory.com/89 [기록보관소]
#    Gate의 개념은 3번에 기술.



# 2.  sigmoid보다 tanh를 사용하는 이유
#     →  sigmoid에 비해 tanh 는 기울기가 ★ 0에서 1 사이 ★ 이므로 Gradient Vanishing problem에 더 강하기 때문.
# ◎ Gradient Vanishing Problem(기울기 소실 문제) 이란?
#     →  https://ydseo.tistory.com/41
#    Gradient Vanishing Problem이란? 인공신경망을 기울기값을 베이스로 하는 method(backpropagation)로 학습시키려고 할 때 발생되는 어려움이다.
#    특히 이 문제는 네트워크에서 앞쪽 레이어의 파라미터들을 학습시키고, 튜닝하기 정말 어렵게 만든다. 이 문제는 신경망 구조에서 레이어가 늘어날수록 더 악화된다.
#     →  https://ydseo.tistory.com/41 [영덕의 연구소]
#    하지만 평소의 DNN구조에서 relu는 기울기가 소실되지 않도록 돕는 역할을 해줄 수 있다.
#     →  http://computing.or.kr/14804/vanishing-gradient-problem%EA%B8%B0%EC%9A%B8%EA%B8%B0-%EC%86%8C%EB%A9%B8-%EB%AC%B8%EC%A0%9C/
# ◎ 0~1의 기울기를 가지는 rnn에 적합한 activation인 tanh을 사용하며(탄젠트 기울기는 0과 1 사이이다. sigmoid도 가능하지만 sigmoid의 미분값은 0~0.25 사이의 값만 표현되어 역전파로 결과값에 대한 가중치 계산 시 전달되는 값이 1/4갑소될 수 있다. 하여 출력 값과 멀어질 수록 학습이 되지 않을 수 있다.) LSTM에서 relu를 사용할 경우 값이 발산해 버릴 수 있다.
#     →  https://muzukphysics.tistory.com/entry/DL-7-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B8%B0%EC%9A%B8%EA%B8%B0%EC%86%8C%EC%8B%A4-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95-Vanishing-Gradient
# ◎ 히든 state의 활성함수(activation function)은 비선형 함수인 하이퍼볼릭탄젠트(tanh)입니다.
#    그런데 활성함수로 왜 비선형 함수를 쓰는걸까요? 밑바닥부터 시작하는 딥러닝의 글귀를 하나 인용해 보겠습니다.
#    선형 함수인 h(x)=cx를 활성 함수로 사용한 3층 네트워크를 떠올려 보세요. 이를 식으로 나타내면 y(x)=h(h(h(x)))가 됩니다. 이 계산은 y(x)=c∗c∗c∗x처럼 세번의 곱셈을 수행하지만 실은 y(x)=ax와 똑같은 식입니다. a=c3이라고만 하면 끝이죠. 즉 히든레이어가 없는 네트워크로 표현할 수 있습니다. 그래서 층을 쌓는 혜택을 얻고 싶다면 활성함수로는 반드시 비선형함수를 사용해야 합니다.
#     →  https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/



# 3. GATE의 개념
# 맨 위에 컨베이너 벨트처럼 흐르는 C값이 cell state이며, LSTM은 이 cell state를 보호하고 컨트롤 하기 위한 세 가지 게이트: forget, input, output gate를 통해 vanishing gradient를 방지하고 그래디언트가 효과적으로 흐를 수 있게 한다.
# ◎ forget gate 
#    f_t는 말그대로 ‘과거 정보를 잊기’위한 게이트다. 시그모이드 함수의 출력 범위는 0 ~ 1 이기 때문에 그 값이 0이라면 이전 상태의 정보는 잊고, 1이라면 이전 상태의 정보를 온전히 기억하게 된다.3
# ◎ input gate 
#    i_t는 ‘현재 정보를 기억하기’위한 게이트다. 이 값은 시그모이드 이므로 0 ~ 1 이지만 hadamard product를 하는 
#    ~    (c대가리 위에 ~임)
#    C_t는 hyperbolic tangent 결과이므로 -1 ~ 1 이 된다.3 따라서 결과는 음수가 될 수도 있다.
# ◎ output gate 
#    O_t는 최종 결과  h_t를 위한 게이트이며, cell state의 hyperbolic tangent를 hadamard product한 값이 LSTM의 최종 결과가 된다.