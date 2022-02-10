import tensorflow as tf
tf.set_random_seed(77)

# 1. 데이터
# to feed dict
x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

# 2. 모델구성
hypothesis = x_train * w + b

# 3-1. 컴파일, 훈련
loss = tf.reduce_mean( tf.square(hypothesis - y_train) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())




loss_val_list = []
w_val_list = []




# for문은 훈련임
for step in range(2001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                        feed_dict={ x_train:x_train_data, y_train:y_train_data })
    if step % 20 == 0:
        print("step:",step, "loss_val:",loss_val, "w_val:",w_val, "b_val:",b_val)

    loss_val_list.append(loss_val)
    w_val_list.append(w_val)

x_test_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_predict = x_test * w_val + b_val
print("[6,7,8] 예측 : ", sess.run(y_predict, feed_dict={ x_test:x_test_data }))

sess.close()

# lr 0.01일때. 0부터 보면 그래프가 너무 급격한 직각그래프가 되므로 100부터로 조정함
import matplotlib.pyplot as plt
plt.plot(loss_val_list[100:])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()