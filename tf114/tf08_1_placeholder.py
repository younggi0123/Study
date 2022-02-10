# Linear1  (ctrl+c+v to) Linear 2


import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]



###################################################################추가내용###################################################################

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

##############################################################################################################################################




W = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

# 2. 모델구성
hypothesis = x_train * W + b

# 3-1. 컴파일, 훈련
loss = tf.reduce_mean( tf.square(hypothesis - y_train) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()                   # 세션 선언
sess.run(tf.global_variables_initializer())     # 세션 초기화

for step in range(2001):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],           # train은 결과치에 대한 반환값이 필요없기에 '_' 공백    # placeholder엔 123 넣음
                        feed_dict={ x_train:[1,2,3], y_train:[1,2,3] })
    if step % 20 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        
        #위에서 한번에 반환값해서 sessrun했으니까
        print(step, loss_val, W_val, b_val)     # 이렇게 하면 sess run 한 번만 하면 되겠다.
        
sess.close()