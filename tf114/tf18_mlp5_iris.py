import numpy as np   
import tensorflow as tf 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
tf.compat.v1.set_random_seed(44)

datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
# print(x_data.shape, y_data.shape) #(150, 4) (150, 1)          # 4
y_data = y_data.reshape(-1,1)

print(y_data)            # 000011112222 도배된상태
print(np.unique(y_data)) #[0 1 2]                               # 3

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# 희소행렬을 OneHot 인코딩을 시키기 위해!  OneHotEncoder(categorical_features= [0], sparse=False) 혹은 .toarray()
encoder = OneHotEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()
# scikit-learn의 OneHotEncoder는 그냥 변수에 객체를 다시 할당해주기만 한다.
# toarray()를 쓰면 array형식의 output을 얻을 수 있는데, dataframe으로 다시 매핑할 수는 없다(기존 dataframe의 shape과 안맞으므로)

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

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w1 = tf.Variable(tf.zeros([4,20]),name='weight1')
b1 = tf.Variable(tf.zeros([20]), name='bias1')
Hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random_normal([20, 16]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([16]), name='weight2')
Hidden_layer2 = tf.sigmoid(tf.matmul(Hidden_layer1, w2) + b2)


# /???왜  10, 1맞춰줘야 되는거지..
w3 = tf.compat.v1.Variable(tf.random_normal([16, 8]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([8]), name='weight3')
Hidden_layer3 = tf.sigmoid(tf.matmul(Hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([8, 3]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random_normal([3]), name='weight4')

hypothesis = tf.nn.softmax( tf.matmul(Hidden_layer3, w4) + b4)

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

# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 1 2 2 2 0 0 0 0 1 0 0 2 1
#  0 0 0 2 1 1 0 0]
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0 0 0 2 1 1 0 0]

# accuracy :  [0.9777778]