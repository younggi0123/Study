# tf07_2  (ctrl+c+v to) tf08_2

# 실습
# 1. [4]        확인용
# 2. [5, 6]     확인용

# 3. [6, 7, 8]로 결과값 빼기

# 위 값들을 이용해서 predict하라. x_test라는 placeholder 생성
# model predict아닌 직접 hypo ㄱㄱ
# 힌트) x_test라는 placeholder생성 hypo 들어가서 sess.run하면 될 것
# Linear1  (ctrl+c+v to) Linear 2


# hypothesis를 sess.run하시오.


import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
x_test = tf.placeholder(tf.float32, shape=[None])
# y_test = tf.placeholder(tf.float32, shape=[None])


w = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

# 2. 모델구성
hypothesis = x_train * w + b

# 3-1. 컴파일, 훈련
loss = tf.reduce_mean( tf.square(hypothesis - y_train) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(loss)
# predict = tf.cast(hypothesis=[4,5,6], dtype=tf.float32)


####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
# 나 ver.

# 3-2. 훈련
sess = tf.compat.v1.Session()                   # 세션 선언
sess.run(tf.global_variables_initializer())     # 세션 초기화

# for문은 훈련임
for step in range(4000):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],           # train은 결과치에 대한 반환값이 필요없기에 '_' 공백    # placeholder엔 123 넣음
                        feed_dict={x_train:[1,2,3], y_train:[1,2,3] })
    if step % 20 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        
        #위에서 한번에 반환값해서 sessrun했으니까
        print(step, loss_val, w_val, b_val)    # 이렇게 하면 sess run 한 번만 하면 되겠다.


test_hypo = x_test*w + b
print( sess.run( test_hypo, feed_dict={ x_test:[4,5,6] }))

# # [3.9948459 4.991861  5.988876 ]
# sess.close()


# # 나는 test_hypo를 만들지 않고 hypothesis를 넣으려고 했는데 그러면 train값이 들어가서 for문(즉, fit)을 돌면서 변질되잖아?
# # 그래서 test식을 만들어 W랑 b 넣어 전제하고 feed_dict에 x_test값을 넣어서 predict 했다.






# https://stackoverflow.com/questions/45761584/how-to-predict-using-trained-tensorflow-model
# Well the 'pred' op is your actual outcome (as it's used to compare with y when calculating the loss), so something like the following should do the trick:
# print(sess.run([pred], feed_dict={x: _INPUT_GOES_HERE_ })








####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
# Y-Ram Version.
# 예를들어 epoch가 커지면서 loss값이 점점 좋아짐을 알 수 있다.
# 반복문을 돌면서 loss값이 예를 들어 0.9를 넘어가고 있다고 하면
# break를 걸며 while문안으로 옮겨준 test_hypo에 그 멈춘 위치의 가중치(W)와 bias(b)를 캐치해서 넣어준다.
# 이후 print 하면 일종의 early stopping과 같은 느낌으로 predict로 넣어준 값과 비슷함을 알 수 있다.


# # 3-2. 훈련

# # test_list = [4]

# with tf.compat.v1.Session() as sess:        # tf.~~~Session()을 sess로써 실행하고 작업이 다 끝나면 종료해라.
# # sess.close()    # session은 항상 열었으면 닫아주어야한다. with문을 쓰면 자동으로 종료된다.
# # sess = tf.compat.v1.Session()
#     sess.run(tf.compat.v1.global_variables_initializer())

#     step = 0
#     while True:
#         step += 1
#         # sess.run(train)     # 여기서 실행이 일어난다.
#         _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
        
#         if step % 20 == 0:
#             # print(f"{step+1}, {sess.run(loss)}, {sess.run(w)}, {sess.run(b)}")
#             print(step, loss_val, w_val, b_val)
        
#         if w_val >= 0.9999999:
            
#             predict = x_test*w+b
            
#             predict = sess.run(predict,feed_dict={x_test:[6,7,8]})
#             print(predict)
#             break
# [6. 7. 8.]


# sess.close()

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################

# 선생님 Ver.
x_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_predict = x_test * w_val + b_val
print("[6,7,8] 예측 : ", sess.run(y_predict, feed_dict={x_test:x_data }))


sess.close()