import tensorflow as tf
tf.compat.v1.set_random_seed(66)

# 5행 3열
x_data = [[73, 51, 65],                         # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79] ]
y_data = [[152], [185], [180], [205], [142] ]   # (5, 1)


x = tf.compat.v1.placeholder(tf.float32, shape = [None, 3] )   # [행무시, 3]     # x데이터 열 3개
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1] )   # 5행 1열

# w = tf.Variable(tf.random.normal([1]), name='weight')          # y = [None, 1] 오류
w = tf.Variable(tf.random.normal([3, 1]), name='weight')         # 행열 계산은 곱하는것1의 행개수 와 곱하는것2의 열개수가 같아야 하는거 알쥐?
# w = tf.Variable(tf.random_normal([3, 1]), name='weight')       # Same

# 행렬곱 # 앞의 열이랑 뒤의 행개수가 맞아야 연산이 되겠쭁?
# [[1,2],                  [[1],
# [3,4],   *   [[1],   =    [3],
# [5,6]]        [0]]        [5]]

# x 열이랑 같은 수로 맞춰줘야함 여기선  x 열은 3임
#   x          w        행렬곱
# (3,2)  *   (2,5)   =  (3, 5)
# (4,3)  *   (3,1)   =  (4, 1)
# (10,7) *   (7,6)   =  (10,6)

# but, bias는 덧셈이라 노상관 !!

b = tf.Variable(tf.random.normal([1]), name='bias')     # y와 매치되므로 5, 1 # 깃참고 # bias는 덧셈이므로 shape변화 없음
# hypothesis = x * w + b
# => 그냥 곱으로 연산하면 안됨!
# 텐서프롤상에서 행렬곱으로 연산해야함
hypothesis = tf.matmul(x, w) + b


# 3-1. 컴파일
loss = tf.reduce_mean( tf.square(hypothesis-y) )            # 현재loss : mse ( (예측값-y)^2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)   # 1 × 10-5를 의미.다시 말해 0.00001  # E가 10을 밑으로 하는 지수를 의미함. 즉 1E-5 = 1*10^-5

train = optimizer.minimize(loss)

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y: y_data })
    # print(step, '\t', loss_v, '\t', w_val )
    if step % 1 == 0:       #20 주면 20단위로 봄
        print(step, loss_val, w_val, b_val)
    
from sklearn.metrics import r2_score, mean_absolute_error
# y_predict =  x * w_val +  b_v
y_predict = tf.matmul(x, w_val) + b_val
y_predict_data = sess.run(y_predict, feed_dict={x:x_data, y: y_data })

print('\ny_predict_data : \n', y_predict_data)

r2 = r2_score(y_data, y_predict_data)
print("r2 : " ,r2)

mae = mean_absolute_error(y_data, y_predict_data)
print("mae : ", mae)

sess.close()

