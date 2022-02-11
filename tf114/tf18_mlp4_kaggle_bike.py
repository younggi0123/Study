import tensorflow as tf
import pandas as pd
tf.set_random_seed(42)

# 1. 데이터
# Data load
path = "./_data/kaggle/titanic/"
path = "./_data/kaggle/bike/"
train = pd.read_csv(path + 'train.csv')
# print(train)        # (10866, 12)
# x & y 설정
x_data = train.drop( ['datetime', 'casual', 'registered', 'count'], axis=1) # 이렇게 4개 빼고 컬럼 구성
y_data = train['count']
print(x_data.shape, y_data.shape)  # (10886, 8) (10886,)

# y_data = y_data.values.reshape(-1,1)
y_data = y_data.values.reshape(10886,1)

# Input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.zeros([8,20]),name='weight')
b1 = tf.Variable(tf.zeros([20]), name='bias')
Hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)


w2 = tf.compat.v1.Variable(tf.random_normal([20, 16]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([16]), name='weight2')
Hidden_layer2 = tf.sigmoid(tf.matmul(Hidden_layer1, w2) + b2)


# /???왜  10, 1맞춰줘야 되는거지..
w3 = tf.compat.v1.Variable(tf.random_normal([16, 8]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([8]), name='weight3')
Hidden_layer3 = tf.sigmoid(tf.matmul(Hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([8, 1]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight4')


#2. 모델 구성
hypothesis = tf.sigmoid(tf.matmul(Hidden_layer3, w4) + b4)


#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w4, b4],
                                         feed_dict = {x : x_data,
                                                      y:y_data})
    if epochs % 100 == 0:
        print(epochs, loss_val,w_val, b_val)

#4. 평가, 예측
y_pred = tf.matmul(x, w4) + b4
y_pred_data = sess.run(y_pred, feed_dict={x : x_data})
print('y_pred_data :',y_pred_data)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data , y_pred_data)
print('r2 :', r2)

sess.close()
 