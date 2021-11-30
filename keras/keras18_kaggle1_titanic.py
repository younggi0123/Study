# 캐글 
# kaggle의 login의 competitions의 titanic-MachineLearningFromDisasterㄱㄱ
# test를 predict해서 submission파일로 빼서 만드는것
# test데이터로 survived를 찾아내서 submission에 넣는것
#  타이타닉 데이터에서 : 결측치처리/이상치처리/string라벨링처리를 못하니까 일단 kaggle2.py 바이크 부터한다.

# 비코 우상단에 edit CSV 버튼 있음
import numpy as np
import pandas as pd

# 1. 데이터
# Data Load
path = "./_data/titanic/"
# train = pd.read_csv(path + "train.csv")
# print(train)
# print(train.shape)      # (891, 12)
# test = pd.read_csv(path + "test.csv")
# print(test)
# print(test.shape)      # (418, 11)
# gender_submission = pd.read_csv(path + "gender_submission.csv")
# print(gender_submission)
# print(gender_submission.shape)      # (418, 2)
# Pandas로 데이터 불러다 쓸 땐 .data, .target 아님, 이건 사이킷런이 제공하는 툴이니까.

train = pd.read_csv( path + "train.csv", index_col=0, header=0 )  # index_column = n번째이며 ,header=0)이 기본인 상태
                                                        # ,header=1 헤더위치 변경, 인덱스 없을경우,header=None으로 가능
gender_submission = pd.read_csv(path + "gender_submission.csv", index_col=0, header=0 )
# Test Data를 predict시키면 결과값이 나오잖아? 근디 우리가 쉅시간에 했던부분은 train을 test와 train로 나눠 훈련시키고
# 캐글이나 대회에서 주는 테스트는 제출용 test 즉 predict용이라 훈련시킬 수 없다. test에 대한 y는 gender_submission이다.
# _data폴더의 gender_submission보면 알겠지만 이 값은 임의로 들어있는 값이다.
# 평소처럼 train에서 train/test로 나눠 실험하고 predict값 구하면 survived이고 이를 입력해주면 되겠다.

# print(test.shape) (418,10)
# print(gender_submession.shape) (418,1)

print(train.describe()) #문자로된..(object는 문자)니까 수치해석이 안됨





#y = pd.get_dummies(y, drop_first=False)   # 열을 n-1개 생성토록 해주는 drop_first// 0으로 가득찬 열이 사라지는 걸 알 수 있다.




# 컬럼명(헤더)은 데이터가 아닌데 데이터로 들어가 버리면 문제이다.
# 즉, train data 의 헤더를 데이터로 안 잡도록 해야 할 것이다.# editCSV header조정부분 조정가능
# https://stackoverflow.com/questions/18175602/how-do-you-delete-the-header-in-a-dataframe
# Survived만 y축이고 나머지는 x축으로 들어갈 것이다.
# CSV 파일의 헤더 인덱스 명확히, shape맞춰서 모델링ㄱㄱ