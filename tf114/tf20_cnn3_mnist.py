import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense

tf.compat.v1.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. 모델구성

# Layer 1
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 128], initializer=tf.contrib.layers.xavier_initializer()) 
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
print(L1_maxpool)  # Tensor("MaxPool2d:0", shape=(?, 14, 14, 128), dtype=float32)
 
# Layer 2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128, 64], initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool =  tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L2_maxpool)  # Tensor("MaxPool2d_1:0", shape=(?, 7, 7, 64), dtype=float32) 

# Layer 3
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 32], initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool =  tf.nn.max_pool2d(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L3_maxpool) # Tensor("MaxPool2d_2:0", shape=(?, 4, 4, 64), dtype=float32)

# Layer 4
w4 = tf.compat.v1.get_variable('w4', shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4_maxpool =  tf.nn.max_pool2d(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L4_maxpool)  # Tensor("MaxPool2d_3:0", shape=(?, 2, 2, 32), dtype=float32)

# Flatten 
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*32])

# Layer 5
w5 = tf.compat.v1.Variable(tf.random.normal([2*2*32, 64], mean=0, stddev=tf.math.sqrt(2/(2*2*32+64)), name='w5'))
b = tf.compat.v1.Variable(tf.zeros([64]), name='bias')
L5 = tf.nn.relu(tf.compat.v1.matmul(L_flat, w5) + b)
L5 = tf.nn.dropout(L5, keep_prob=0.7)

w6 = tf.compat.v1.Variable(tf.random.normal([64, 10], mean=0, stddev=tf.math.sqrt(2/(64+10)), name='w6'))
b2 = tf.compat.v1.Variable(tf.zeros([10], name='bias1'))

output = tf.nn.softmax(tf.matmul(L5, w6) + b2)


training_epochs = 7
batch_size = 100
total_batch = int(len(x_train)/batch_size)

# 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(output), axis=1)) # categorical_crossentropy

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss = 0
        
    for i in range(total_batch):        # 600번
        start =  i * batch_size         # 0 
        end = start + batch_size        # 100
        batch_x, batch_y = x_train[start:end], y_train[start:end]  # 100개씩 나눈 train
            
        feed_dict = {x:batch_x, y:batch_y}
            
        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        
        avg_loss = batch_loss / total_batch
    
    print('Epoch : ', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))

prediction = tf.equal(tf.arg_max(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print("ACC : ", sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# ACC :  0.9884
