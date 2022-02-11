# [실습] 히든레이어 2개 이상 늘리기 !


import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터
x_data = [[0, 0], [0,1], [1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

# Input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# MLP
# 연산량 늘리기 => 노드수 늘리기 !

# 2. 모델구성
# 상위 레이어의 아웃풋에 이어서 구성
# Original Code ( before this source )
# w = tf.compat.v1.Variable(tf.random.normal([2,1]), name='weight1')
# b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias1')

w1 = tf.compat.v1.Variable(tf.random.normal([2,10]), name='weight1')
b1 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias1')
Hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random_normal([10, 5]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([5]), name='weight2')
Hidden_layer2 = tf.sigmoid(tf.matmul(Hidden_layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random_normal([5, 2]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([2]), name='weight3')
Hidden_layer3 = tf.sigmoid(tf.matmul(Hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([2, 1]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight4')

hypothesis = tf.sigmoid(tf.matmul(Hidden_layer3, w4) + b4 )



# 3-1. 컴파일
cost =   - tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

# optimizer = tf.train.AdamOptimizer(learning_rate=0.0000011)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000011)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)

train = optimizer.minimize(cost)    # same as cost

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(10001):
    loss_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 10 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)

#평가 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y ), dtype=tf.float32))

pred, acc = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print(#"에측값 : \n", hy_val, 
      # "\n predict : \n", pred,    # 너무 길어서 생략
        "\n Accuracy :", acc)

from sklearn.metrics import r2_score, accuracy_score
# accs = accuracy_score(y_test, predicted)
# print(accs)

sess.close()




# GradientDescentOptimizer(learning_rate=0.08)
# epochs 10001

# Accuracy : 1.0