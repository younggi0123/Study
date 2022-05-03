# 비지도 학습이란?  =>     y가 없는것 입니다 ! ( like PCA & transform으로 바꿔주는 애들 )
## 군집모델/ in 분류모델
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_iris()

# data , target이 방식은 사이킥런에서 편의상 제공하는 것임
# x = datasets.data
# print(type(x))        # <class 'numpy.ndarray'>
# y = datasets.target

irisDF = pd.DataFrame(datasets.data, columns= [datasets.feature_names])
print(irisDF)
#     sepal length (cm) sepal width (cm) petal length (cm) petal width (cm)
# 0                 5.1              3.5               1.4              0.2
# 1                 4.9              3.0               1.4              0.2
# 2                 4.7              3.2               1.3              0.2
# 3                 4.6              3.1               1.5              0.2
# 4                 5.0              3.6               1.4              0.2
# ..                ...              ...               ...              ...
# 145               6.7              3.0               5.2              2.3
# 146               6.3              2.5               5.0              1.9
# 147               6.5              3.0               5.2              2.0
# 148               6.2              3.4               5.4              2.3
# 149               5.9              3.0               5.1              1.8
# [150 rows x 4 columns]

kmeans = KMeans(n_clusters=3, random_state=66)  # 군집선을 3개 그어 3범위로 나누겠다.
                                                # 아이리스 데이터는 y가 3개였으니 여기서도 3으로 set해서 해보겠다
kmeans.fit(irisDF)

print(kmeans.labels_)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 1 1 1 1 2 1 1 1 1
#  1 1 2 2 1 1 1 1 2 1 2 1 2 1 1 2 2 1 1 1 1 1 2 1 1 1 1 2 1 1 1 2 1 1 1 2 1
#  1 2]

print(datasets.target)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
# y값을 지정하지 않았음에도 target값이 거의 유사하게 나옴을 볼 수 있다.



'''
n_clusters : int, optional, default: 8
  k개(분류할 수) = 클러스터의 중심(centeroid) 수

init : {‘k-means++’, ‘random’ or an ndarray}
  Method for initialization, defaults to ‘k-means++’:
  ‘k-means++’
  ‘random’
  np.array([[1,4],[10,5],[16,2]]))  

n_init : int, default: 10
  큰 반복 수 제한 (클러스터의 중심(centeroid) 위치)

max_iter : int, default: 300
  작은 반복수 제한

tol : float, default: 1e-4
  inertia (Sum of squared distances of samples to their closest cluster center.) 가 tol 만큼 줄어 들지 않으면 종료 (조기 종료)
'''




# 컬럼 기존 4개인데 
# 2개를 더 컬럼명 만들어서 넣어주려함

# cluster는 predict개념
irisDF['cluster'] = kmeans.labels_# 새로 생성된 놈
irisDF['target'] = datasets.target# 기존 y값

# 약간 이런느낌 ▼
# pred_y = kmeans.labels_          
# real_y = datasets.target
# print(accuracy_score(real_y,pred_y))


# iris_result = irisDF.groupby( ['target', 'cluster'] )['sepal_length'].count()
# print(iris_result)


# accuracy score 도출 !


print("accuracy score : ", round(accuracy_score(irisDF['target'], irisDF['cluster']), 4 ) )
# accuracy score :  0.8933


# 최대 반복 max_iter=300,












# # print(accuracy_score(datasets.target,kmeans.labels_))


# from sklearn.metrics import silhouette_score
# from sklearn.metrics import adjusted_mutual_info_score
# from sklearn.metrics import adjusted_rand_score
# from sklearn.metrics import completeness_score
# from sklearn.metrics import fowlkes_mallows_score
# from sklearn.metrics import homogeneity_score
# from sklearn.metrics import mutual_info_score
# from sklearn.metrics import normalized_mutual_info_score
# from sklearn.metrics import v_measure_score


# print(silhouette_score(x['cluster'], x['target'])) #0.6322939531368102 #실루엣 계수: 군집간 거리는 멀고 군집내 거리는 가까울수록 점수 높음 (0~1), 0.5 보다 크면 클러스터링이 잘 된거라 평가
# print(adjusted_mutual_info_score(datasets.target, y_predict)) #1.0
# print(adjusted_rand_score(datasets.target, y_predict)) #1.0
# print(completeness_score(datasets.target, y_predict)) #1.0
# print(fowlkes_mallows_score(datasets.target, y_predict)) #1.0
# print(homogeneity_score(datasets.target, y_predict)) #1.0
# print(mutual_info_score(datasets.target, y_predict)) #1.077556327066801
# print(normalized_mutual_info_score(datasets.target, y_predict)) #1.0
# print(v_measure_score(datasets.target, y_predict)) #1.0