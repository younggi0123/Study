# 아웃라이어 처리

# (아웃라이어 기준은?)
# 우선 중간값 잡고 IQR을 하여, +- 150 % 까지 경계로 잡는다.

# ex2) -100, 1, 2, 3, 4, 5, 6, 10만
# 중위값은 4이고 1사분위는 2이고 3사분위는 6이다.

# 6-2 = 4 4의 1.5배는 6
# 1사분위-6, 3사분위+6 즉, -4 ~ 12까지가 outlier의 경계이다.
import numpy as np

aaa = np.array([1,2, -1000, 4, 5, 6, 7, 8, 90, 100, 500, 12, 13])   #중위수 6.5

# [[ 아웃라이어 함수 적용 ]]
def outliers(data_out):
    quantile_1, q2, quantile_3 = np.percentile(data_out, [25,50,75])
    print("1사분위 : ", quantile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quantile_3)
    iqr = quantile_3 - quantile_1
    print("iqr : ", iqr)
    lower_bound = quantile_1 - (iqr * 1.5)
    upper_bound = quantile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) |        #  이 줄 또는( | )
                    (data_out<lower_bound))         #  아랫줄일 경우 반환

print( outliers(aaa) )
# 1사분위 :  4.0
# q2 :  7.0
# 3사분위 :  13.0
# iqr :  9.0
# (array([ 2,  8,  9, 10], dtype=int64),)
# 10-4 = 6   ,  6*1.5 = 9

outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)
#  (array([ 2,  8,  9, 10] 위치인덱스

# 아웃라이어경계1 : 4-9= -5, 아웃라이어경계2: 13+9 = 22

# 시각화
# 실습
# boxplot
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(5,10))
boxplot = sns.boxplot(data=aaa, color="red")
plt.show()