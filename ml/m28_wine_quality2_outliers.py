from catboost import train
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# 아웃라이어, 라벨개수, F1 스코어 확인해보았는가?

# 1. 데이터
# 데이터 불러오기
path = "../_data/kaggle/wine/"
# read_csv takes a sep param, in your case just pass sep=';' like so:
# data = read_csv(csv_path, sep=';')
datasets = pd.read_csv(path+"winequality-white.csv", index_col=None, header=0, sep=';')#첫째줄이 헤더고 헤더가 있음
# index_col's default is 'None'

print( datasets.head() )
print( datasets.describe() ) # 수치데이터에 한해서 확인키 편하다.     # .DESCR은 사이킥런 샘플용 예제 한정(실무용은 다 csv니까 describe겠지?)
print( datasets.info() )    #11개 컬럼 다 나온다.
                            # quality 만 int이고 y값으로 뺄 것이다. 
                            # 모든 데이터가 not-null이다. = 결측치 없다!

# 왜 넘파이화?   더 빠르니까! (왜인지는 검색해보자)
# 넘파이화 하여보자.!
datasets = datasets.values
print(type(datasets))
print(datasets.shape)
# pandas에서 x는 드랍했었지
# x = datasets.drop(['quality'], axis =1)
# pandas에서 y는 그 컬럼만 빼왔었지
# y = datasets['quality']

# 넘파이에서는??
# 모든행, 10번쨰 열까지
x = datasets[: , :11]
y = datasets[: , 11]

print("라벨 : ", np.unique(y, return_counts=True))
# 라벨 :  (array([  3.,  4.,  5.,   6.,    7.,   8.,     9.]), 
#          array([  20,  163, 1457, 2198,  880,  175,    5 ] , dtype=int64))
# 3이 20개 4가 163개 ....



import matplotlib.pyplot as plt
################################### 아웃라이어 확인 ###################################


def boxplot_vis(data, target_name):
    plt.figure(figsize=(30, 30))
    for col_idx in range(len(data.columns)):
        # 6행 2열 서브플롯에 각 feature 박스플롯 시각화
        plt.subplot(6, 2, col_idx+1)
        # flierprops: 빨간색 다이아몬드 모양으로 아웃라이어 시각화
        plt.boxplot(data[data.columns[col_idx]], flierprops = dict(markerfacecolor = 'r', marker = 'D'))
        # 그래프 타이틀: feature name
        plt.title("Feature" + "(" + target_name + "):" + data.columns[col_idx], fontsize = 20)
    # plt.savefig('../figure/boxplot_' + target_name + '.png')
    plt.show()
boxplot_vis(datasets,'white_wine')

def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier




# def outliers(data_out):
#     try:
#         i=0
#         while True:
#             # 반복설정
#             i=i+1

#             if data_out[:,i-1:i] is not None :
#                quantile_1, q2, quantile_3 = np.percentile(data_out[:,i-1:i], [25,50,75])
#                print(i,"열")
#                print("1사분위 : ", quantile_1)
#                print("q2 : ", q2)
#                print("3사분위 : ", quantile_3)
               
#                iqr = quantile_3 - quantile_1
#                print("iqr : ", iqr)
#                print("\n")
#                lower_bound = quantile_1 - (iqr * 1.5)
#                upper_bound = quantile_3 + (iqr * 1.5)

#             else:
#                 return np.where((data_out[:,i-1:i] > upper_bound) |        #  이 줄 또는( | )
#                             (data_out[:,i-1:i] < lower_bound))         #  아랫줄일 경우 반환
#     except Exception:
#         pass


# # 시각화
# # 실습
# # boxplot
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(5,10))
# boxplot = sns.boxplot(data=x, color="red")
# plt.show()

################################### 아웃라이어 처리 ###################################






#스케일러
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
# f1 score?  남자 90대 여자 10이면 대부분 남자라고 하면 acc가 90%잖아? 
# => 이렇게 데이터가 불균형한 문제를 대처하기 위해서 f1_score 사용한다.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8, stratify=y)
# stratify = y는 yes의 y가 아닌 y의 y임!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(n_jobs=-1)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_predict = model.predict(x_test)

# accuracy 확인법 1.
acc_score = model.score(x_test, y_test)
print('model.score : ', acc_score)                                  # model.score :  0.6591836734693878
# accuracy 확인법 2.
print('acc_score : ', accuracy_score(y_test, y_predict))            # acc_score :  0.6591836734693878
# F1_score
print('f1_score : ', f1_score(y_test, y_predict, average='macro') ) # f1_score :  0.41005452777318885
# 다중분류 f1_score은 macro로 돌려야
print('f1_score : ', f1_score(y_test, y_predict, average='micro') ) # f1_score :  0.6591836734693878
