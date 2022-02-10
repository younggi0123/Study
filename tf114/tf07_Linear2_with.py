# Linear1  (ctrl+c+v to) Linear 2


import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]
W = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)


# 2. 모델구성
hypothesis = x_train * W + b

# 3-1. 컴파일, 훈련
loss = tf.reduce_mean( tf.square(hypothesis - y_train) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

# 3-2. 훈련
# compat.v1 오류뜨지 말라고 써줌
with tf.compat.v1.Session() as sess:    # sess = tf.session() 과 같음
                                        # with 쓰면 with문이 끝나는 시점에서 자동으로 sess.close가 된다(session 끝내려고 쓰는거)
                                        # as로 sess라고 정의한거임
    # sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))
# sess.close()

