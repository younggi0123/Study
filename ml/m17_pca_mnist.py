import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

# PCA를 하며 y를 빼버림(없어도 되는이유 앞서 필기)
# 공백처리 '_'부분을 안 가져오겠다 (y값 뺌)
(x_train, _ ), (x_test, _ ) = mnist.load_data()

# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

# append로 행으로 붙임
x = np.append( x_train, x_test, axis=0 )

# print(x.shape)
# (70000, 28, 28)

x = np.reshape(x,(70000,784))
##############################################################
# 실습
# PCA를 통해 0.95 이상인 n_components가 몇 개인지 도출하시오.
# 0.95, 0.99, 0.999, 1.0
# np.argmax 쓰기!
##############################################################

pca = PCA(n_components=705)
x = pca.fit_transform(x)
# print(x.shape)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)

# print(np.argmax(cumsum,axis=0)) # 704

# If a numbur of n_components are 784
# 1.0000000000000022

# If a numbur of n_components are 705


result_val = [0.95, 0.99, 0.999, 1.0]
for i in result_val:
    print(i,"=>",np.argmax(cumsum>i))
# 개수 출력
# 0.95 => 153
# 0.99 => 330
# 0.999 => 485
# 1.0 => 0


# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# # plt.plot(pca_EVR)
# plt.grid()
# plt.show()
