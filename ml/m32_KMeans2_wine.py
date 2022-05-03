# 비지도 학습이란?  =>     y가 없는것 입니다 ! ( like PCA & transform으로 바꿔주는 애들 )
## 군집모델/ in 분류모델
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_wine()

# data , target이 방식은 사이킥런에서 편의상 제공하는 것임
# x = datasets.data
# print(type(x))        # <class 'numpy.ndarray'>
# y = datasets.target

wineDF = pd.DataFrame(datasets.data, columns= [datasets.feature_names])
print(wineDF)
#     alcohol malic_acid   ash alcalinity_of_ash magnesium total_phenols flavanoids nonflavanoid_phenols proanthocyanins color_intensity   hue od280/od315_of_diluted_wines proline
# 0     14.23       1.71  2.43              15.6     127.0          2.80       3.06                 0.28            2.29            5.64  1.04                         3.92  1065.0
# 1     13.20       1.78  2.14              11.2     100.0          2.65       2.76                 0.26            1.28            4.38  1.05                         3.40  1050.0
# 2     13.16       2.36  2.67              18.6     101.0          2.80       3.24                 0.30            2.81            5.68  1.03                         3.17  1185.0
# 3     14.37       1.95  2.50              16.8     113.0          3.85       3.49                 0.24            2.18            7.80  0.86                         3.45  1480.0
# 4     13.24       2.59  2.87              21.0     118.0          2.80       2.69                 0.39            1.82            4.32  1.04                         2.93   735.0
# ..      ...        ...   ...               ...       ...           ...        ...                  ...             ...             ...   ...                          ...     ...
# 173   13.71       5.65  2.45              20.5      95.0          1.68       0.61                 0.52            1.06            7.70  0.64                         1.74   740.0
# 174   13.40       3.91  2.48              23.0     102.0          1.80       0.75                 0.43            1.41            7.30  0.70                         1.56   750.0
# 175   13.27       4.28  2.26              20.0     120.0          1.59       0.69                 0.43            1.35           10.20  0.59                         1.56   835.0
# 176   13.17       2.59  2.37              20.0     120.0          1.65       0.68                 0.53            1.46            9.30  0.60                         1.62   840.0
# 177   14.13       4.10  2.74              24.5      96.0          2.05       0.76                 0.56            1.35            9.20  0.61                         1.60   560.0
# [178 rows x 13 columns]

kmeans = KMeans(n_clusters=3, random_state=66)  # 군집선을 3개 그어 3범위로 나누겠다.
                                                # 아이리스 데이터는 y가 3개였으니 여기서도 3으로 set해서 해보겠다
kmeans.fit(wineDF)

print(kmeans.labels_)
# [1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 0 1 1 0 1 1 1 1 1 1 0 0
#  1 1 0 0 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 2 0 2 2 0 2 2 0 0 0 2 2 1
#  0 2 2 2 0 2 2 0 0 2 2 2 2 2 0 0 2 2 2 2 2 0 0 2 0 2 0 2 2 2 0 2 2 2 2 0 2
#  2 0 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 0 2 2 0 0 0 0 2 2 2 0 0 2 2 0 0 2 0
#  0 2 2 2 2 0 0 0 2 0 0 0 2 0 2 0 0 2 0 0 0 0 2 2 0 0 0 0 0 2]

print(datasets.target)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
# y값을 지정하지 않았음에도 target값이 거의 유사하게 나옴을 볼 수 있다.










# cluster는 predict개념
wineDF['cluster'] = kmeans.labels_# 새로 생성된 놈
wineDF['target'] = datasets.target# 기존 y값





print("accuracy score : ", round(accuracy_score(wineDF['target'], wineDF['cluster']), 4 ) )
# accuracy score :  0.1854


