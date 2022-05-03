import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1,5)
y = softmax(x)

ratio = y
labels = y

plt.pie(ratio, labels, shadow=True, startangle=90)
plt.show()


# So, softmax의 전체 합은 1이다 !
# activation의 주목적은 '한정'하는것
# 활성화함수를 통과한 값은 값이지나갈때 제한한다 제한하지 않으면
# 커지기도하고 음수화되는 문제도 있으니까 이러한 문제를
# relu로 음수제한 sigmoid로 0과 1사이로 한정하는 등으로 해결 ㄱ



# 1/ .......+ e^n-2  + e^n-1 + e^n