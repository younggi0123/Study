import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

# 1. 데이터
# 8 ,4
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7] ]
# 8 ,3
y_data = [[0, 0, 1],        #2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],        #1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],        #0
          [1, 0, 0]]
# x * w = y
# (8,  4)*(4  ,3) => (8, 3)

# x_predict = [[1, 11, 7, 9]]     # (1, 4) => (N, 4)


# 2. 모델구성
# 실습 ㄱㄱ
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 4] )
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 3] )

w = tf.Variable(tf.random.normal([4, 3]), name='weight')
b = tf.Variable(tf.random.normal([1, 3]), name='bias')  # 출력노드 개수에 맞춰서 ! (덧셈으로 맞춰준거)
                                                        # y의 출력 개수만큼 출력 (행렬의 덧셈방법-행과 열의 갯수가 같아야함)
                                                        # 더해지는건 한개인데 나가는게(y 개수가) 3개라서 1,3

# BASIC : hypothesis = x * w + b
# NeuralNetwork
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) # model.add(Dense(3, activation='softmax'))

# 3-1. 컴파일
############################################## Categorical_Crossentropy
loss = tf.reduce_mean( - tf.reduce_sum(y * tf.log(hypothesis), axis=1)  )

# #이전 버전 
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
# train = optimizer.minimize(loss)

#합친 버전 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, "loss : ", loss_val)
            
            
    # # Predict
    # results = sess.run(hypothesis, feed_dict={x:x_predict} )
    # print(results, sess.run(tf.arg_max(results, 1)))

    # predict = results
    results = sess.run(hypothesis, feed_dict={x:x_data} )# x_data 넣어서 predict생성해서 ㄱㄱ
    print(results, sess.run(tf.arg_max(results, 1)))
###########################################################추가#####################################################################
                                        # 추가로 accuracy_score 로직 넣어줄 것 !!
    accuracy = tf.reduce_mean( tf.cast( tf.equal(y_data, results), dtype=tf.float32 ) )

    print( 'accuracy :', sess.run(accuracy))

sess.close()

