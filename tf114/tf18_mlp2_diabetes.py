from sklearn.datasets import load_diabetes
import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터
datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)  # (442, 10) (442,)
y_data = y_data.reshape(442,1)
print(x_data.shape, y_data.shape)  # (442, 10) (442,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7, shuffle = True, random_state=42)

# Input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 2. 모델구성
w1 = tf.compat.v1.Variable(tf.random.normal([10,8]), name='weight1')
b1 = tf.compat.v1.Variable(tf.random.normal([8]), name='bias1')
Hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random_normal([8, 5]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([5]), name='weight2')
Hidden_layer2 = tf.sigmoid(tf.matmul(Hidden_layer1, w2) + b2)


# /???왜  10, 1맞춰줘야 되는거지..
w3 = tf.compat.v1.Variable(tf.random_normal([5, 10]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([10]), name='weight3')
Hidden_layer3 = tf.sigmoid(tf.matmul(Hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight4')


#2. 모델 구성
hypothesis = tf.matmul(Hidden_layer3, w4) + b4

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-6)
optimizer = tf.train.AdamOptimizer(learning_rate=0.8) #아담
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w4, b4],
                                         feed_dict = {x : x_data,
                                                      y:y_data})
    if epochs % 200== 0:
        print(epochs, loss_val)
        
#4. 평가, 예측
y_pred = tf.matmul(x, w4) + b4
y_pred_data = sess.run(y_pred, feed_dict={x : x_data})
# print('y_pred_data :',y_pred_data)

from sklearn.metrics import r2_score, mean_absolute_error
y_predict = sess.run(hypothesis,{x : x_test})
r2 = r2_score(y_test , y_predict)
print('r2 :', r2)


# r2 : -0.0031280735633829604