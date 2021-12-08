#  model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10, 10, 1 ) ))
# total parameter가 50인 이유는 무엇인가? 
'''
[컨볼루션 레이어의 학습 파라미터 수는 “입력채널수X필터폭X필터높이X출력채널수“로 계산됩니다]
첫 번째 conv2d 에서는 (10, 10, 1) 이미지에 (2, 2) 필터를 10개 사용하였습니다.
이 때, 어떻게 파라미터의 갯수가 50개가 나올 수 있을까?
먼저 (2, 2) 필터 한개에는 2 x 2 = 4개의 파라미터가 있습니다.
컬러는 입력되는 3-channel 각각에 서로 다른 파라미터들이 입력 되므로 R, G, B 에 해당하는 3이, 흑백은 1이.
그리고 Conv2D(10, ...) 에서의 10는 10개의 필터를 적용하여 다음 층에서는 채널이 총 10개가 되도록 만든다는 뜻입니다.
여기에 bias로 더해질 상수가 각각의 채널 마다 존재하므로 5개가 추가로 더해지게 됩니다.
정리하면, 2 x 2(필터 크기) x 1 (#입력 채널(RGB)) x 10(#출력 채널) + 10(출력 채널 bias) = 50이 됩니다.
'''
# Conv2D페러미터 설명
# tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
# taewan.kim/post/cnn
# https://justkode.kr/deep-learning/pytorch-cnn
'''
Parameters
일단 Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')의 파라미터는 다음과 같습니다.

in_channels: 입력 채널 수을 뜻합니다. 흑백 이미지일 경우 1, RGB 값을 가진 이미지일 경우 3 을 가진 경우가 많습니다.
out_channels: 출력 채널 수을 뜻합니다.
kernel_size: 커널 사이즈를 뜻합니다. int 혹은 tuple이 올 수 있습니다.
stride: stride 사이즈를 뜻합니다. int 혹은 tuple이 올 수 있습니다. 기본 값은 1입니다.
padding: padding 사이즈를 뜻합니다. int 혹은 tuple이 올 수 있습니다. 기본 값은 0입니다.
padding_mode: padding mode를 설정할 수 있습니다. 기본 값은 'zeros' 입니다. 아직 zero padding만 지원 합니다.
dilation: 커널 사이 간격 사이즈를 조절 합니다. 해당 링크를 확인 하세요.
groups: 입력 층의 그룹 수을 설정하여 입력의 채널 수를 그룹 수에 맞게 분류 합니다. 그 다음, 출력의 채널 수를 그룹 수에 맞게 분리하여, 입력 그룹과 출력 그룹의 짝을 지은 다음 해당 그룹 안에서만 연산이 이루어지게 합니다.
bias: bias 값을 설정 할 지, 말지를 결정합니다. 기본 값은 True 입니다.
'''
'''
[filters필터]
미지의 특징을 찾아내기 위한 공용 파라미터입니다. Filter를 Kernel이라고 하기도 합니다. CNN에서 Filter와 Kernel은 같은 의미입니다. 필터는 일반적으로 (4, 4)이나 (3, 3)과 같은 정사각 행렬로 정의됩니다. CNN에서 학습의 대상은 필터 파라미터 입니다. <그림 1>과 같이 입력 데이터를 지정된 간격으로 순회하며 채널별로 합성곱을 하고 모든 채널(컬러의 경우 3개)의 합성곱의 합을 Feature Map로 만듭니다. 필터는 지정된 간격으로 이동하면서 전체 입력데이터와 합성곱하여 Feature Map을 만듭니다.
[kernel_size커널(=필터)]
strides


'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D # 1D는 선만 그어. 2D부터 이미지
from tensorflow.python.keras.layers.core import Dropout
# Convolution은 레이어로 구성되어 있다.
# 참고!!gaussian37.github.io/dl-keras-number-of-cnn-param/
model  =  Sequential() 
# 2x2로 잘라본다            # 4차원
model.add(Conv2D(10, kernel_size=(2,2), strides=1, padding='same', input_shape=(10, 10, 1 ) )) # 이미지를 받았을때도  어떻게 자를건진 본인이 결정한다.(height,width,rgb값)
# model.add(Conv2D(10, kernel_size=(2,2), strides=5, padding='valid', input_shape=(6, 6, 1 ) ))   #strides 성큼성큼 걷다(1일 경우 stride만큼 걷는것.)
                                                                                                # default는 1이다. 
# 수집 정제 시 shape가 같아야 한다.                                 # Conv2D => (9,9,10)  10이 마지막 노드 값으로 들어가 필터 개수가 정해진다
                                                                                   # padding='same' 패딩으로 감싸서 가장자리 데이터도 정밀하게 해줌

# model.add(MaxPooling2D())               # maxpooling도 레이어이다.#전달이 어떻게 되는지는 summary찍어본다.(10,10,10) to (5,5,10)
                                          # parameters :  pool_size=(2,2)   pool_size=2(정수 가능)  pooling에 사용할 filter의 크기를 정함(단순한 정수, 또는 튜플형태 (N,N))
model.add(Conv2D(5, (2,2), activation="relu") )      # 위의 아웃풋과 차원은 같아야 하므로 인풋으로 4차원으로 받음.

                                                                # Conv2D => (7,7,5)
model.add(Dropout(0.2))
model.add(Conv2D(7, (2,2), activation="relu") )                 # Conv2D => (6,6,7)
model.add(Flatten()) # 노드에서 들어온 정보의 shape만 관여하기에 데이터 연산은 없음
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))
model.summary() 
                # Convolution 역시 여러번 할 수 있다.
#(N, 6x6x7) 순서도 바뀌지않아.  (None,252)와 같이 일렬로 펴준다. 이때 변환수치를 굳이 기입할 필요가 있을까?
# Convolution layer와 Dense layer를 엮기위한 시도=> Reshape로 계속 하는 것보다는 가로로 펴주는 기능이 더 효율적이겠다.(펴준거 그대로 다음 레이어로 전달하면 되니까.)
# layer상에서 하는 작업.
# ★Flatten(import)★ 차원형태의 레이어를 일렬로 펴준다.


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 4, 4, 10)          50
# =================================================================
# Total params: 50
# Trainable params: 50
# Non-trainable params: 0
# _________________________________________________________________


model.add(Flatten())

model.summary()





# 3x3로 잘라본다         #과제 :명칭이 뭔지(10,, 3, 5, 등등 숫자에 대한 명칭)   # keras.io 들어가봐
# model.add(Conv2D(10, kernel_size=(3,3), input_shape=(0, 10, 1 ) )) #inputsize - kernelsize + 1   # 8, 8, 10
# model.add(Conv2D(5, (3,3), activation="relu") ) 
# model.add(Conv2D(7, (3,3), activation="relu") )
# model.summary() 

# 윤호 창민 유천 재중 준수->
# LabelEncoder
# 0, 1, 2, 3, 4     => (5, )        => (5,1)
                    # [0,1,2,3,4]   # [ [0],[1],[2],[3],[4] ]

# Dense layer 는 통상 2차원에서 주로 사용한다. 가로x세로 행렬 convolution layer와 엮어 마지막 레이어를 dense1의 activation softmax나 sigmoid등 알아서.
# covolution과 dense를 엮어만 주면 하단은 알아서 구성이 가능하다 dense는 차원을 2차원을 받으니까


