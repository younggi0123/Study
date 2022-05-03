# 아웃라이어 처리

# import numpy as np

# aaa = np.array( [
#                 [1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
#                 [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]
#                 ] ) # 2행 13열

# # (2, 13) => (13, 2)
# aaa = np.transpose(aaa)
# 컬럼이 둘 이상인 행렬형태(13, 2)라면 ?! 

# 실습 : 다차원의 outlier가 출력되도록 수정하기!
# 아래 함수의 data_out에 다차원을 받을 수 있게 수정하여 사용한다.


# # [[ 아웃라이어 함수 적용 ]]
# def outliers(data_out):
#     quantile_1, q2, quantile_3 = np.percentile(data_out, [25,50,75])
#     print("1사분위 : ", quantile_1)
#     print("q2 : ", q2)
#     print("3사분위 : ", quantile_3)
#     iqr = quantile_3 - quantile_1
#     print("iqr : ", iqr)
#     lower_bound = quantile_1 - (iqr * 1.5)
#     upper_bound = quantile_3 + (iqr * 1.5)
#     return np.where((data_out>upper_bound) |        #  이 줄 또는( | )
#                     (data_out<lower_bound))         #  아랫줄일 경우 반환

# print( "1행 : ")
# line1 = outliers( aaa[:1, :] )
# print(line1)
# # 1행 : 
# # 1사분위 :  25.75
# # q2 :  50.5
# # 3사분위 :  75.25
# # iqr :  49.5

# print("\n")

# print( "2행 : ")
# line2 = outliers( aaa[1:2, :] )
# print(line2)
# # 2행 :
# # 1사분위 :  51.5
# # q2 :  101.0
# # 3사분위 :  150.5
# # iqr :  99.0


# # 1사분위 :  6.25
# # q2 :  64.5
# # 3사분위 :  475.0
# # iqr :  468.75

# outliers_loc = outliers(aaa)
# print("이상치의 위치 : ", outliers_loc)
# #  (array([ 2,  8,  9, 10] 위치인덱스

# # 아웃라이어경계1 : 4-9= -5, 아웃라이어경계2: 13+9 = 22

# # 시각화
# # 실습
# # boxplot
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(5,10))
# boxplot = sns.boxplot(data=aaa, color="red")
# plt.show()








# 본래 1차원만 받는 데이터를 다차원으로 받을 수 있도록.
# [[ 아웃라이어 함수 데이터 프레임 화 ]]


import numpy as np

aaa = np.array( [
                [1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]
                ] ) # 2행 13열

# (2, 13) => (13, 2)
aaa = np.transpose(aaa)

print(aaa)
def outliers(data_out):
    try:
        i=0
        while True:
            # 반복설정
            i=i+1

            if data_out[:,i-1:i] is not None :
               quantile_1, q2, quantile_3 = np.percentile(data_out[:,i-1:i], [25,50,75])
               print(i,"열")
               print("1사분위 : ", quantile_1)
               print("q2 : ", q2)
               print("3사분위 : ", quantile_3)
               
               iqr = quantile_3 - quantile_1
               print("iqr : ", iqr)
               print("\n")
               lower_bound = quantile_1 - (iqr * 1.5)
               upper_bound = quantile_3 + (iqr * 1.5)

            else:
                return np.where((data_out[:,i-1:i] > upper_bound) |        #  이 줄 또는( | )
                            (data_out[:,i-1:i] < lower_bound))         #  아랫줄일 경우 반환
    except Exception:
        pass

print( outliers(aaa) )




################################### 아웃라이어 확인 ###################################
# 출처 : 세종형님

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



