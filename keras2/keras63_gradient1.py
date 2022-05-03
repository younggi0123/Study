import numpy as np
import matplotlib.pyplot as plt

# 이차함수 그래프
f = lambda x: x**2 - 4*x + 6

x = np.linspace(-1, 6, 100)

print(x, len(x))
# [-1.         -0.92929293 -0.85858586 -0.78787879 -0.71717172 -0.64646465
#  -0.57575758 -0.50505051 -0.43434343 -0.36363636 -0.29292929 -0.22222222
#  -0.15151515 -0.08080808 -0.01010101  0.06060606  0.13131313  0.2020202
#   0.27272727  0.34343434  0.41414141  0.48484848  0.55555556  0.62626263
#   0.6969697   0.76767677  0.83838384  0.90909091  0.97979798  1.05050505
#   1.12121212  1.19191919  1.26262626  1.33333333  1.4040404   1.47474747
#   1.54545455  1.61616162  1.68686869  1.75757576  1.82828283  1.8989899
#   1.96969697  2.04040404  2.11111111  2.18181818  2.25252525  2.32323232
#   2.39393939  2.46464646  2.53535354  2.60606061  2.67676768  2.74747475
#   2.81818182  2.88888889  2.95959596  3.03030303  3.1010101   3.17171717
#   3.24242424  3.31313131  3.38383838  3.45454545  3.52525253  3.5959596
#   3.66666667  3.73737374  3.80808081  3.87878788  3.94949495  4.02020202
#   4.09090909  4.16161616  4.23232323  4.3030303   4.37373737  4.44444444
#   4.51515152  4.58585859  4.65656566  4.72727273  4.7979798   4.86868687
#   4.93939394  5.01010101  5.08080808  5.15151515  5.22222222  5.29292929
#   5.36363636  5.43434343  5.50505051  5.57575758  5.64646465  5.71717172
#   5.78787879  5.85858586  5.92929293  6.        ] 100

y = f(x)

################# 그리기
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
