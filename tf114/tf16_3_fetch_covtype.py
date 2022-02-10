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
# 희소행렬을 OneHot 인코딩을 시키기 위해!  OneHotEncoder(categorical_features= [0], sparse=False) 혹은 .toarray()
encoder = OneHotEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7, shuffle = True, random_state=42)

#스케일러를 했을때 값이 더 좋아짐
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

# 공간창출
x = tf.placeholder(tf.float32, shape=[None, 54])
y = tf.placeholder(tf.float32, shape=[None, 7])

w = tf.Variable(tf.random.normal([54,7]), name = 'weight') 
# (x열 , y열) -> x데이터와 (N, 54) * (54 , 3) -> (None ,7 )이 되기때문에

b = tf.Variable(tf.random.normal([1,7]), name = 'bias') 
#y의 출력 갯수만큼 출력 (행렬의 덧셈방법-행과 열의 갯수가 같아야함)
#더해지는건 1나인데 나가는게(y갯수가) 7개라서 1,7

#소프트맥스
hypothesis = tf.nn.softmax( tf.matmul(x, w) + b )

#카테고리칼 크로스 엔트로피 
loss = tf.reduce_mean( -tf.reduce_sum(y * tf.log(hypothesis), axis=1) )
# categorical_crossentropy

#이전 버전 
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#합친 버전 
# train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

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

# [0 1 1 ... 0 0 1]
# [0 1 1 ... 1 1 1]
# accuracy :  [0.7164666]

sess.close()