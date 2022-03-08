# 18-8 mnist에 dropout을 추가함

# import tensorflow as tf

# from tf114.tf18_mlp1_boston import Hidden_layer1
 
# x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
# y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# w1 = tf.compat.v1.Variable(tf.random.normal([2, 30]), name='weight')
# b1 = tf.compat.v1.Variable(tf.random.normal([30]), name='bias')

# Hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) +b1)
# layers =  tf.nn.dropout(Hidden_layer1, keep_porb=0.7)   # 70% 쓸꼬얌

# # 예전에 쓰던 방식 (from tensorflow와 같은 것임)
# # keras 로딩해서 쓰면 좀 느릴 수 있다
# from keras.datasets import mnist
# from keras.layers import Dense
# from keras.models import Sequential

# datasets = mnist.load_data()




# 다층레이어 구성

#여러가지 activation 구성 가능 
# hypothesis = tf.nn.relu(tf.matmul(x_train, w3) + b3) 
# hypothesis = tf.nn.elu(tf.matmul(x_train, w3) + b3) 
# hypothesis = tf.nn.selu(tf.matmul(x_train, w3) + b3)
# hypothesis = tf.sigmoid(tf.matmul(x_train, w3) + b3) 
# hypothesis = tf.nn.dropout(layer1, keep_prob=0.3) 
# hypothesis = tf.nn.softmax(tf.matmul(x_train, w3) + b3) 


#실습 
# from tensorflow.keras.datasets import mnist #기존 방식 
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) - 얘는 4차원을 받는데 3차원이라서 차원을 늘려야한다 
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,) 

#train만 원핫인코더 해줌 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()

print(y_train.shape) #(60000, 10)
print(y_test[:5])

x_train = x_train.reshape(60000,28*28) # 스케일링 쉐이프와 dense의 2차원 쉐이프 맞춰주기 
x_test = x_test.reshape(10000,28*28) 

#스케일링
from sklearn.preprocessing import StandardScaler,MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = StandardScaler()
# scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)

#모델구성 
#2. 모델구성 
x = tf.compat.v1.placeholder(tf.float32, shape=[None,28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

#히든레이어 1 
w1 = tf.Variable(tf.random.normal([28*28,256], stddev=0.1)) 
#[데이터의 열의수, 내가 주고싶은 노드의수] 
# stddev는 랜덤수의 범위를 정해주는 것. 
# default는 1.0이다 0.1로 해주면 적은 범위의 수들이 랜덤하게 뽑힌다.
# 초반의 weight를 잡기위해서 stddev를 집어넣은것.
b1 = tf.Variable(tf.random.normal([256], stddev=0.1))

layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)  #새로운 x값이 탄생했다고 생각하고 layer1을 다음 x 자리에 추가
layer1 = tf.nn.dropout(layer1, keep_prob=0.1) 
#keep_prob 남겨놓을 파라미터수를 % 형태로 지정해놓은것 

#히든레이어 2
w2 = tf.Variable(tf.random.normal([256,124], stddev=0.1)) #[8,10] = [이전노드의 열, 내가 주고싶은 노드의수]  
b2 = tf.Variable(tf.random.normal([124], stddev=0.1))
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2) 
layer2 = tf.nn.dropout(layer2, keep_prob=0.1) 

#히든레이어 3
w3 = tf.Variable(tf.random.normal([124,62], stddev=0.1)) #[8,10] = [이전노드의 열, 내가 주고싶은 노드의수]  
b3 = tf.Variable(tf.random.normal([62], stddev=0.1))
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3) 

#activation을 relu로 지정하고 진행                           

#아웃풋 레이어 
w4 = tf.Variable(tf.random.normal([62,10], stddev=0.1))
b4 = tf.Variable(tf.random.normal([10], stddev=0.1))
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4) 


#카테고리칼 크로스 엔트로피 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) 

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00002)
train = optimizer.minimize(loss)


sess = tf.compat.v1.Session() 
sess.run(tf.global_variables_initializer())

# predict = 

for epochs in range(100):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
            feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0 :
        print(epochs, "loss : ", cost_val, "\n", hy_val)

#평가 예측
y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
y_pred = np.argmax(y_pred, axis= 1)
y_test = np.argmax(y_test, axis= 1)

print(y_pred)
print(y_test)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), dtype=tf.float32))
a = sess.run([accuracy])
print('accuracy : ',a)


sess.close()

# [0 0 0 ... 0 0 0]
# [7 2 1 ... 4 5 6]
# accuracy :  [0.098]
