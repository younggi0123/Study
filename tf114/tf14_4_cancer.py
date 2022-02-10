import tensorflow as tf
from sklearn.datasets import load_breast_cancer
tf.compat.v1.set_random_seed(66)
# 1. 데이터

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target

print(x_data.shape, y_data.shape)  #(569, 30) (569,)
y_data = y_data.reshape(-1,1)


# 2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 30] )
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1] )

w = tf.Variable(tf.random.normal([30, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


hypothesis = tf.sigmoid(tf.matmul(x, w) + b)


# 3-1. 컴파일
loss =   - tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 10 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)

y_predict = tf.cast( hypothesis > 0.5, dtype=tf.float32 )

print( sess.run( hypothesis>0.5, feed_dict = {x:x_data, y:y_data} ) )
print( sess.run(tf.equal(y,y_predict), feed_dict={x:x_data, y:y_data}))


accuracy = tf.reduce_mean( tf.cast(tf.equal(y, y_predict), dtype=tf.float32) )

pred, acc = sess.run([y_predict, accuracy],feed_dict={x: x_data, y:y_data})

print("=========================================================")
print("예측값 : \n", hy_val)
print("예측결과 : \n", pred)
print("Accuracy : ", acc)

sess.close()