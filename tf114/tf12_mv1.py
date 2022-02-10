import tensorflow as tf
tf.set_random_seed(66)

# 3 columns( 3 features ), 5rows
#         첫번  둘번  세번  네변  다섯번
x1_data = [73.,  93.,  89.,  96.,  73.]     # 국어
x2_data = [80.,  88.,  91.,  98.,  66.]     # 영어
x3_data = [75.,  93.,  90.,  100., 70.]     # 수학
y_data = [152.,  185., 180., 196., 142.]    # 환산점수

# x는 (5, 4), y는 (5, 1) 또는 (5,)
# x에다 w곱해야 행렬곱이니까 현실에서 쓸때야 y= w1x1 . . . 이겠지만
# y= w1 x1 + w2 x2  .  . . 를 y= x1 w1  + x2 w2 ...로 해야.

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 스칼라 한 개짜리에 데이터를 넣어준다.
w1 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight3')
b = tf.compat.v1.Variable(tf.random_normal([1]), name='bias' )
# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run( [w1, w2, w3] ))


# 2. 모델
# Linear Regressor
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

# 3-1. 컴파일
loss = tf.reduce_mean( tf.square(hypothesis-y) )            # 현재loss : mse ( (예측값-y)^2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)   # 1 × 10-5를 의미.다시 말해 0.00001  # E가 10을 밑으로 하는 지수를 의미함. 즉 1E-5 = 1*10^-5

train = optimizer.minimize(loss)

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


for step in range(21):
    _, loss_v, w_val1, w_val2, w_val3, b_v = sess.run([train, loss, w1, w2, w3, b], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y: y_data })
    print(step, '\t', loss_v, '\t', w_val1,'\t', w_val2,'\t', w_val3 )


from sklearn.metrics import r2_score, mean_absolute_error
y_predict =  x1*w_val1 + x2*w_val2 + x3*w_val3 + b_v

y_predict_data = sess.run(y_predict, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y: y_data })
print('y_predict_data : ', y_predict_data)

r2 = r2_score(y_data, y_predict_data)
print("r2 : " ,r2)

mae = mean_absolute_error(y_data, y_predict_data)
print("mae : ", mae)

sess.close()



# 평가지표 r2, mse

# y_predict_data :  [158.97856 179.49509 182.99956 197.91934 135.02621]
# r2 :  0.9333083213631421
# mae :  4.87523193359375