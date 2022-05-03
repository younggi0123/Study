# Activation
# 레이어의 최종과정을 통과하는 함수이다.
# 리니어는 패스   :   명시하지 않으면 linear(명시하지 않아도 리니어는 동일한 값이니 의미 없음)이므로 (y= wx+b자체가 linear이니) 정리하지 않았다
# Activation설명 : https://hwk0702.github.io/ml/dl/deep%20learning/2020/07/09/activation_function/
# 텐서플로우 공홈 : https://www.tensorflow.org/api_docs/python/tf/keras/activations
# relu계열 함수화 : https://gooopy.tistory.com/56
# ★★ 액티베이션 전체 함수화 :  https://subinium.github.io/introduction-to-activation/



# 찾아서 정리
# elu , Selu, Leaky relu
# 3_2,  3_3,   3_4
# 셀루성능 좋음





# 시그모이드

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )   # 배열의 지수함수 https://wooono.tistory.com/214
                                    # y  =  1  / ( 1 + (e^-x) )
x = np.arange(-5, 5, 0.1)

print(len(x))

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()

