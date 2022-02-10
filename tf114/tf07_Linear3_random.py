# Linear2  (ctrl+c+v to) Linear 3


import tensorflow as tf
tf.set_random_seed(42)

# 1. 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]
# W = tf.Variable(1, dtype=tf.float32)
# b = tf.Variable(1, dtype=tf.float32)
# placeholder로 데이터 작업해야게쮜?? placeholder = 입력값(feed_dict), 변수는 연산 위한 변하는 값!

###################################################################추가내용###################################################################
# W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
# b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

W = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(W))  # W 하나를 출력하기 위해 위의 두줄이나 써줘야 했다... ㅠ
                      # randomseed 66일때 : [0.06524777]
                      # randomseed 42일때 : [0.8021455]


##############################################################################################################################################



# 2. 모델구성
hypothesis = x_train * W + b

# 3-1. 컴파일, 훈련
loss = tf.reduce_mean( tf.square(hypothesis - y_train) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))
# sess.close()

