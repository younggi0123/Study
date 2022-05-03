import numpy as np
import matplotlib.pyplot as plt

# Leaky ReLU를 만들어보자
def Leaky_ReLU(x):
     return np.maximum(0.01*x, x)

x = np.arange(-5, 5, 0.1)
y = Leaky_ReLU(x)

plt.plot(x, y)
plt.grid()
plt.show()


# 찾아서 정리
# elu , Selu, Leaky relu
# 3_2,  3_3,   3_4
# 셀루성능 좋음