# 이진분류!
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
tf.compat.v1.set_random_seed(66)
# 1. 데이터

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target

print(x_data.shape, y_data.shape)  #(569, 30) (569,)
y_data = y_data.reshape(-1,1)      #(569, 30) (569,1)

# split ㄱㄱ
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7,random_state=9)


# 2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 30] )
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1] )

w1 = tf.Variable(tf.zeros([30,8]),name='weight')
b1 = tf.Variable(tf.zeros([8]), name='bias')
Hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)


w2 = tf.compat.v1.Variable(tf.random_normal([8, 5]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([5]), name='weight2')
Hidden_layer2 = tf.sigmoid(tf.matmul(Hidden_layer1, w2) + b2)


# /???왜  10, 1맞춰줘야 되는거지..
w3 = tf.compat.v1.Variable(tf.random_normal([5, 3]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([3]), name='weight3')
Hidden_layer3 = tf.sigmoid(tf.matmul(Hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([3, 1]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight4')

hypothesis = tf.sigmoid(tf.matmul(Hidden_layer3, w4) + b4)


# 3-1. 컴파일
loss =   - tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

# optimizer = tf.train.AdamOptimizer(learning_rate=0.0000011)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000011)

train = optimizer.minimize(loss)    # same as cost

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 100 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)

#평가 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y ), dtype=tf.float32))

pred, acc = sess.run([predicted, accuracy], feed_dict={x:x_test, y:y_test})

print("어큐 스코어 : ",acc)


sess.close()

# 어큐 스코어 :  0.3625731