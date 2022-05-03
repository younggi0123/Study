# 람다함수를 통한 epochs당 weight값(기울기) 도출
# x제곱 - 4x  +  6의 기울기 0값인 # 2x2 지점을 찾아내고자 한다.


import numpy as np

# 이차함수
f = lambda x: x**2 - 4*x + 6

# x = 4
# print(f(x))


# 기울기 = 가중치(weight)
gradient = lambda x: 2*x - 4


# 람다를 쓰는이유?
# 파이썬식 문법으로 함수를 간단히 짧게 줄여쓰려고!
# 예시.
# def  f(x):
#     temp = x**2 - 4*x +6
#     return temp

# x = 5
# gradient(x)


# 미분값 -> 기울기 weight ->가중치


# 최적의 weight 찾기 !



########################################################################################################################################################


x = 0.0             # 초기값 설정
epochs = 20
learning_rate = 0.5

print("step\t x\t f(x)")
print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(0, x, f(x)))            # 초기값 ( 0에포일때 x값과 y값 )


                                                                    # 포문으로 에포 돌면서
for i in range(epochs):
    x = x - learning_rate * gradient(x)
    
    print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(  i+1   , x, f(x)))  # epoch i회
    print("")
    


# step 01
# step 02 (2 epoch)
# x  = 1빼기 0.25 곱하기  (gradient x가 1이므로 2 * 1 - 4 = -2 )  그러므로 아래와 같다.
# 1 - 0.25 * -2 = 1+0.5 = 1.5
# 1.5^2  = 2.25-6+6 = 2.25

