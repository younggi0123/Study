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

##############################################################
# 실습
# PCA를 통해 0.95 이상인 n_components가 몇 개인지 도출하시오.
# 0.95, 0.99, 0.999, 1.0
# np.argmax 쓰기!
##############################################################

# x = np.reshape(x,(70000,784))
x = np.reshape(x, (x.shape[0], (x.shape[1]*x.shape[2])) )

pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
# print(cumsum)

print(np.argmax(cumsum >= 0.95)+1 )     # 154
print(np.argmax(cumsum >= 0.99)+1 )     # 331
print(np.argmax(cumsum >= 0.999)+1 )     # 486
print(np.argmax(cumsum) +1)    # 712로 찍히는데 0부터 시작이니 713   #1.0의 시작지점