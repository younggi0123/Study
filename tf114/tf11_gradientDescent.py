import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.set_random_seed(66)  # 값 계속 변하는것 방지

x_train = [1,2,3]
y_train= [1,2,3]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# w값 random
# w = tf.compat.v1.Variable(tf.random_normal([1]), name='weight')

# w값 지정
w = tf.compat.v1.Variable(0, dtype=tf.float32)


hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))


# O p t i m i z e r
# lr = 0.1
lr = 0.21

gradient = tf.reduce_mean(( w * x -y ) * x)
descent = w - lr * gradient

update = w.assign(descent)              # w = w - lr * gradient # w에 descent할당(으로 쭉쭉 update하며 이어지는것)
                                        # w로 업데이트

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

for step in range(21):
    # sess.run(update, feed_dict = { x:x_train, y:y_train })
    # print(step, '\t', sess.run(loss, feed_dict={ x:x_train, y:y_train }), sess.run(w) )

    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()

print("================================ W history ===============================")
print(w_history)
print("============================== loss history ==============================")
print(loss_history)
print("==========================================================================")
plt.plot(w_history, loss_history)
plt.xlabel('Weight')
plt.ylabel('loss')
plt.show()



# 가중치 지정하고 lr =0.21 주면 5epoch만에 끝나는걸 볼 수있다

# 0        4.6666665       0.97999996
# 1        0.0018666765    0.9996
# 2        7.466442e-07    0.999992
# 3        2.9579894e-10   0.9999998
# 4        1.2908193e-13   1.0
# 5        0.0     1.0
# 6        0.0     1.0
# 7        0.0     1.0
# 8        0.0     1.0
# 9        0.0     1.0
# 10       0.0     1.0
# 11       0.0     1.0