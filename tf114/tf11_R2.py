import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.set_random_seed(66)  # 값 계속 변하는것 방지

x_train_data = [1,2,3]
y_train_data = [1,2,3]
x_test_data = [4,5,6]
y_test_data = [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
# y_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

 
 # w값 random
w = tf.compat.v1.Variable(tf.random_normal([1]), name='weight')

# w값 지정
# w = tf.compat.v1.Variable(0, dtype=tf.float32)


hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))


# O p t i m i z e r
lr = 0.1
# lr = 0.21

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

    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train_data, y:y_train_data})
    print(step, '\t', loss_v, '\t', w_v)


####################################### 실습 - R2 만들기 !!#######################################
from sklearn.metrics import r2_score, mean_absolute_error
# 1. predict만덜어서 ytest와 비교


# b = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')

# x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
# predict = x_test_data * w_v + b

# # total_error = tf.reduce_sum(tf.square(tf.sub(y, tf.reduce_mean(y))))
# # unexplained_error = tf.reduce_sum(tf.square(tf.sub(y_test_data, predict)))
# # R_squared = tf.sub(1, tf.div(unexplained_error, total_error))
# x_data = [6,7,8]
# x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
# y_predict = x_test * w_v + b_val
# print("[6,7,8] 예측 : ", sess.run(y_predict, feed_dict={x_test:x_data }))


# from sklearn.metrics import r2_score,mean_absolute_error
# R2 = r2_score(y_test_data, predict, multioutput='variance_weighted')
# print( R2 )


y_predict = x_test * w_v # bias는 없음
y_predict_data = sess.run(y_predict, feed_dict={x_test:x_test_data})
print('y_predict_data : ', y_predict_data)

r2 = r2_score(y_test_data, y_predict_data)
print("r2 : " ,r2)

mae = mean_absolute_error(y_test_data, y_predict_data)
print("mae : ", mae)

sess.close()

# y_predict_data :  [3.9999924 4.9999905 5.9999886]
# r2 :  0.9999999998599378
# mae :  9.5367431640625e-06