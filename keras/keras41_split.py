# split 함수를 선언함 (주가 (종가 시가)등에도 응용 가능하고 몇일씩 자를지 등등 다양하게 응용가능)

import numpy as np

a = np.array(range(1, 11))  # 기본 배열 제공
size = 5                    # 커팅할 사이즈

def split_x(dataset, size):                     # 함수 선언(알고리즘)
    aaa = []                                    # 빈 리스트 선언
    for i in range( len(dataset) - size + 1):   # length함수(dataset만큼 - 자를길이 + 1)    => dataset까지 만큼 for문 돌면 역전되어서 [9:5] 같이 되어버리니까 안 되겠지?
        subset = dataset[i : (i+size)]          # 부분집합( 1줄 ) = dataset 중 1부터 5까지 채워넣겠다.
        aaa.append(subset)                      # 빈 리스트 aaa에 subset을 붙이겠다.
    return np.array(aaa)                        # aaa값을 리턴해서 다음 for문을 진행. 다시 2~6, 3~7 ㄱㄱ

dataset = split_x(a, size)

print(dataset.shape)
print(dataset)          # (6, 5)    6행 5열
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
 '''

#            ▼          # 3차원으로 리쉐잎

dataset=dataset.reshape(2, 3, 5)
print(dataset)          # (2, 3, 5) 2축 3행 5열
'''
[[[ 1  2  3  4  5]
  [ 2  3  4  5  6]
  [ 3  4  5  6  7]]

 [[ 4  5  6  7  8]
  [ 5  6  7  8  9]
  [ 6  7  8  9 10]]]
 '''
 
 
bbb= split_x(a,size)

print(bbb)
print(bbb.shape)          # (6, 5)    6행 5열

x= bbb[:, :-1]
y= bbb[:, -1]
print(x,y)
print(x.shape,y.shape)

#  [:, : ] #        :하나면 모든행.     :두개면 모든행+모든열