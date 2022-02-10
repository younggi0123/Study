# y = wx + b

# 행렬연산이기에 사실은 이렇다.
# y = x * w + b


# https://datamod.tistory.com/164
# https://datamod.tistory.com/164
# https://datamod.tistory.com/164
# https://datamod.tistory.com/164









import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)


# 2. 모델구성
hypothesis = x_train * W + b        # y= wx + b
# ⓧ   ⓦ   ⓑ
#   \ /    /
#   (*)   /
#     (+)     #여기까지 hypothesis

# 3-1. 컴파일, 훈련
loss = tf.reduce_mean( tf.square(hypothesis - y_train) )    # mse 공식 (여기서 root 씌우면  rmse일 것)
                                                            # 계산 234 -123 = 111 => 제곱해도 111 => reduce_mean 1+1+1/3 = 1 => 로스값은 1 이다.
                                                            # square => 배열 원소의 제곱값 범용 함수 https://rfriend.tistory.com/300
                                                            # reduce_mean 배열전체 원소합을 평균 https://webnautes.tistory.com/1235

                                                            # hypothesis = [1,2,3]*1 +1 = [2,3,4]
                                                            # reduce_mean( square  ( [2,3,4] - [1,2,3] )
                                                            # reduce_mean( [1^2, 1^2, 1^2] )
                                                            # 3 / 3 = 1


                                                            

# 그래프 연산
#  ▼ input ▼
# ⓧ   ⓦ   ⓑ
#   \ /    /
#   (*)   /
#     (+)   ⓨ
#         (*)            # 이어서
#            ( ** )
#                (mean)
#                    (min)
#                (train)  (opti)
#                 └Session

#  최적의 weight 최소의 loss

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01) # 다행히 optimizer는 정의되어 있다.
# 케라스에선 별도의 minimizer를 정해주지 않지만( 내장이니까 기본적으로 로스 최소값(최적값)을 구하는. )여기선 해야함.
train = optimizer.minimize(loss)    #optimizer를 최소화하는데 기준은 loss값이다.
                                    # 실제 optimizer = 'sgd' 이 한줄임
                                    # 케라스 model.compile(loss='mse', optimizer='sgd')

# 3-2. 훈련
sess = tf.compat.v1.Session()                   # 세션 선언
sess.run(tf.global_variables_initializer())     # 세션 초기화

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))   # train같은 fix된 값 말고 opt, loss같이 바뀐값에대해 출력만 해주는것

# sess.run에 무엇이 들어가는가? 갱신되는 것은 무엇인가?

