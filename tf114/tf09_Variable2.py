import tensorflow as tf
tf.compat.v1.set_random_seed(66)



# [실습]  (08-1과 동일)

##############################################################################################################
#                                       [09-1의 방법1~3으로 실습진행]                                        #
##############################################################################################################

########################################################################## 방법 1. Session() // sess.run(변수) 
# tf08_3  (ctrl+c+v to) tf09_2
tf.set_random_seed(77)

# 1. 데이터
# to feed dict
x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

# 2. 모델구성
hypothesis = x_train * w + b

# 3-1. 컴파일, 훈련
loss = tf.reduce_mean( tf.square(hypothesis - y_train) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# for문은 훈련임
for step in range(100):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                        feed_dict={ x_train:x_train_data, y_train:y_train_data })
#    if step % 20 == 0:
#       print("step:",step, "loss_val:",loss_val, "w_val:",w_val, "b_val:",b_val)



x_test_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_predict = x_test * w_val + b_val
print("[6,7,8] 예측 : ", sess.run(y_predict, feed_dict={ x_test:x_test_data }))

sess.close()


# w 초기값을 주나 랜덤값을 주나 상관없으니 위를 신경 안 써도 되는 부분이다.
########################################################################## 방법 2. Session() // 변수.eval(session=sess)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = y_predict.eval(session=sess, feed_dict={ x_test:x_test_data })         # 변수.eval
print("[6,7,8] 예측 : ", bbb)
sess.close()



########################################################################## 방법 3. InteractiveSession() // 변수.eval()
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = y_predict.eval(feed_dict={ x_test:x_test_data })
print("[6,7,8] 예측 : ", ccc)
sess.close()
