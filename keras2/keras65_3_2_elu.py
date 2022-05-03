
import numpy as np
import matplotlib.pyplot as plt
def ELU(x):
    return np.maximum( 2*(np.exp(x) - 1) * abs(x)/-x , x )

x = np.arange(-5, 5, 0.1)
y = ELU(x)

plt.plot(x, y)
plt.grid()
plt.show()


# 찾아서 정리
# elu , Selu, Leaky relu
# 3_2,  3_3,   3_4
# 셀루성능 좋음