from sklearn.datasets import load_diabetes
import tensorflow as tf
tf.set_random_seed(42)

#1. 데이터
datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)  # (442, 10) (442,)
y_data = y_data.reshape(442,1)
print(x_data.shape, y_data.shape)  # (442, 10) (442,1)

# 1 dimension vs 2 dimension
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7, shuffle = True, random_state=9)
#r2 score를 위해 train- test 셋구분 


x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

# tf.Variable: 그래프를 계산하면서 최적화 할 변수들입니다. 이 값이 바로 신경망을 좌우하는 값들입니다.
# tf.random_normal: 각 변수들의 초기값을 정규분포 랜덤 값으로 초기화합니다.
w = tf.Variable(tf.random.normal([10,1]), name = 'weight')
b = tf.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델 구성
hypothesis = tf.matmul(x, w) + b 

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-6)

optimizer = tf.train.AdamOptimizer(learning_rate=0.8) #아담
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(200001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                         feed_dict = {x : x_data,
                                                      y:y_data})
    if epochs % 2000== 0:
        print(epochs, loss_val,)
        

#4. 평가, 예측
y_pred = tf.matmul(x, w) + b
y_pred_data = sess.run(y_pred, feed_dict={x : x_data})
# print('y_pred_data :',y_pred_data)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data , y_pred_data)
print('r2 :', r2)

# r2 : 0.5690900495452118

# r2 : 0.5177494226017805