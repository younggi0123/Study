# ML 03_1~  iris를 사용한 실습 진행

from distutils.log import Log
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Data
datasets = load_iris()
# print(datasets.DESCR)
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data   #numpy data
y = datasets.target

print(x)
print(y)
print(type(x))

# df = pd.DAtaFrame(x, columns=datasets['feature_names']) #이건 아랫줄과 같다.
#    =  df = pd.DAtaFrame(x, columns=datasets.feature_names)

# 컬럼명 넣어 프레임 만들어 주기
df = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
print(df)

# y까지 컬럼을 추가해준다
df[ 'Target(Y)' ] = y
print(df)


print("===============================상관계수 히트맵==================================")
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()