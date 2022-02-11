import numpy as np
from sklearn.datasets import fetch_covtype
import tensorflow as tf 
from sklearn.model_selection import train_test_split

tf.compat.v1.set_random_seed(44)
datasets = fetch_covtype()

x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape) #(581012, 54) (581012,)                 # 54
y_data = y_data.reshape(-1,1)

print(y_data)   # 111222333444555666777 도배된상태
print(np.unique(y_data)) # [1 2 3 4 5 6 7]                                # 7

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
encoder = OneHotEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7, shuffle = True, random_state=42)

#스케일러
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
#scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x = tf.placeholder(tf.float32, shape=[None, 54])
y = tf.placeholder(tf.float32, shape=[None, 7])

w1 = tf.Variable(tf.zeros([54,20]),name='weight1')
b1 = tf.Variable(tf.zeros([20]), name='bias1')
Hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random_normal([20, 16]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([16]), name='weight2')
Hidden_layer2 = tf.sigmoid(tf.matmul(Hidden_layer1, w2) + b2)


# /???왜  10, 1맞춰줘야 되는거지..
w3 = tf.compat.v1.Variable(tf.random_normal([16, 8]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([8]), name='weight3')
Hidden_layer3 = tf.sigmoid(tf.matmul(Hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([8, 7]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random_normal([7]), name='weight4')

hypothesis = tf.nn.softmax( tf.matmul(Hidden_layer3, w4) + b4 )
loss = tf.reduce_mean( -tf.reduce_sum(y * tf.log(hypothesis), axis=1) )

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session() 
sess.run(tf.global_variables_initializer())

for epochs in range(5000):
    loss_val, _ = sess.run([loss, train],
            feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0 :
        print(epochs, "loss : ", loss_val)



# #평가 예측
y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
y_pred = np.argmax(y_pred, axis= 1)
y_test = np.argmax(y_test, axis= 1)

print(y_pred)
print(y_test)

accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), dtype=tf.float32))
a = sess.run([accuracy])

print('accuracy : ',a)

sess.close()


# [0 1 1 ... 0 1 1]
# [0 1 1 ... 1 1 1]
# accuracy :  [0.7999071]