# Elliptic Envelope
# 가우스 분산 데이터 세트에서 특이 치를 탐지하기위한 객체

import numpy as np

aaa = np.array( [
                [1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]
                ] ) # 2행 13열

# (2, 13) => (13, 2)
aaa = np.transpose(aaa)
# print(aaa)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)  # 0.1에서 0.2 0.3 갈수록 범위 바깥쪽부터 오염도 순으로 찾아줌
# contaminationfloat, default=0.1 # The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Range is (0, 0.5].
# https://runebook.dev/ko/docs/scikit_learn/modules/generated/sklearn.covariance.ellipticenvelope

outliers.fit(aaa)
results = outliers.predict(aaa)
#######제거 results = outliers.fit_predict(aaa)

print(results)
# import sklearn
# print(sklearn.__version__) # 0.24.2     # 1버전대랑 출력 다름/.



# from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.15)
# pred = outliers.fit_predict(aaa)
# print(pred.shape) # (13,)

# b = list(pred)
# print(b.count(-1))
# index_for_outlier = np.where(pred == -1)
# print('outier indexex are', index_for_outlier)
# outlier_value = aaa[index_for_outlier]
# print('outlier_value :', outlier_value)

# list_aaa = []
# i=0
# end=1
# for list_aaa in range [i, end]:
#     if :
#     bbb = outliers.fit(aaa)
#     list_aaa = np.append(list_aaa,[bbb],axis=0)
#     i=i+1
#     end=end+1
#     np.where(  aaa[i:end,:]  )


# results = outliers.predict(list_aaa)

# print(results)
# import sklearn
# print(sklearn.__version__) # 0.24.2     # 1버전대랑 출력 다름/.



# outliers.fit(aaa)
# results = outliers.predict(aaa)

# def ellipticenvelope(list_aaa):
#     try:
#         i=0
#         print(0)
#         while True:
#             # 반복설정
#             i=i+1
#             print(1)
#             # bbb = np.array([])
#             if list_aaa[i-1:i, :] is not None :
#                print(2)
#                outliers.fit( list_aaa[i-1:i, :] )
#                print(3)
#                result = outliers.predict(list_aaa)
#                print(result)
#                list_aaa = np.append(list_aaa, [result], axis=0)
#                print(4)
#             else:
#                 print(5)
#                 return         #  아랫줄일 경우 반환
#     except Exception:
#         pass


#내가 하다 실패한거
def ellipticenvelope(list_aaa):#함수선언
    i=0
    while True:
        i=i+1
        print(1)
        # 빈리스트가 나올때까지 반복하면서
        if list_aaa[:,i-1:i] != [] :
            bbb = np.array([  ])
            result=outliers.fit_predict( list_aaa[:, i-1:i]  )
            # result값을 빈 어레이에 밑에 더해주려는 것
            bbtg = np.append(bbb, result, axis=0)
        else:
            return bbtg
print( ellipticenvelope(aaa) )






# import numpy as np

# aaa = np.array( [
#                 [1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
#                 [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]
#                 ] ) # 2행 13열
# outliers = EllipticEnvelope(contamination=.1) # 그냥 이런 함수가 있는거임(아웃라이어를 -1로 잡아주는 함수)
# outliers.fit(aaa)
# results= outliers.predict(aaa)
# print(results)