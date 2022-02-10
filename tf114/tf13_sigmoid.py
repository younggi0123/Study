import tensorflow as tf
tf.compat.v1.set_random_seed(66)

# 1. 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]       # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                         # (6, 1)
# weight = 2 by 1,  bias = 1( 노상관 )

# 2. 모델구성
# 실습 ㄱㄱ


x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2] )
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1] )

w = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


# hypothesis = x * w + b
# hypothesis = tf.matmul(x, w) + b
# # model = tf.matmul(x, w ) + b   # 원래 모델로쓰는데 혼동갈까봐 이렇게 안쓰는 중

# # hypothesis를 sigmoid로 통과시켜준다. (활성화함수)
# hypothesis = tf.sigmoid(hypothesis)
#                ▼

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)        # model.add(Dense(1, activation = 'sigmoid'))


# 3-1. 컴파일
# loss = tf.reduce_mean( tf.square(hypothesis-y) )      # mse

# 공식의 minus 유의 !!
loss =   - tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary cross entrophy (시그모이드 !)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
train = optimizer.minimize(loss)
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    # _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y: y_data })
    
    # loss_val, hyp_val, w_val, b_val, _ = sess.run([ loss, hypothesis, w, b, train ],        # 순서변경= train이 나중에 들어가도록.
    #                                 feed_dict={x:x_data, y: y_data })
    
    # if epochs % 200 == 0: #10번만 찍히도록
    #     print(epochs, 'loss : ', loss_val, '\n', hyp_val)
        
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 10 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)


# 200 epoch당 실행의 결과가 6개가 나왔다.
# 원래대로면 000 111나와야겠지만(hypothesis가 프레딕한 값이니..)


# 4. 평가, 예측
# 기준 0.5이고 # cast로 0.5 보다 크면 1.0으로 통과시켜줌
y_predict = tf.cast( hypothesis > 0.5, dtype=tf.float32 )
# tf.cast함수  (= 안의 조건문이 참인지 거짓인지를 판단함)
# 텐서를 새로운 형태로 캐스팅하는데 사용한다.
# 부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다.
# Boolean형태인 경우 True이면 1, False이면 0을 출력한다.
# 쉽게 말해 if True=1 else 0과 같음

# 위의 식에서 반환되는값이 False False False True True True인 것을 float로 바꿔주겠다는말이지?
# y_predict는 0. 0. 0. 1. 1. 1.로 나올 것이다.
# dtype을 int32로 바꾸면 0 0 0 1 1 1 로 나올 것이다.
# float로 바꾼건 y랑 비교할때 y데이터를 placeholder로 넣을때 기본 float32이니까 int32와 비교하면 형별이 다르니 비교할 수가 없다.
# 즉, 비교할때 float vs int 즉, 1. 과 1을 비교하게되면 이는 다른 것이니까 통일시킨것.
print( sess.run( hypothesis>0.5, feed_dict = {x:x_data, y:y_data} ) )
print( sess.run(tf.equal(y,y_predict), feed_dict={x:x_data, y:y_data}))




# equal함수로 y와 y_predict를 비교 한다.
# # cast로 감싸서 동일하면 1 아니면 0
# reducemean으로 평균내준다

accuracy = tf.reduce_mean( tf.cast(tf.equal(y, y_predict), dtype=tf.float32) )

pred, acc = sess.run([y_predict, accuracy],feed_dict={x: x_data, y:y_data})

print("=========================================================")
print("예측값 : \n", hy_val)
print("예측결과 : \n", pred)
print("Accuracy : ", acc)

sess.close()

# 싸이킥런 으로 비교해보기.  ( sess run통과한 값에서 바꿔)


# from sklearn.metrics import r2_score, mean_absolute_error
# # y_predict = tf.matmul(x, w_val) + b_val
# y_predict = tf.matmul(x,w) + b

# y_predict_data = sess.run(y_predict, feed_dict={x:x_data, y: y_data })

# print('\ny_predict_data : \n', y_predict_data)

# r2 = r2_score(y_data, y_predict_data)
# print("r2 : " ,r2)

# # mae = mean_absolute_error(y_data, y_predict_data)
# # print("mae : ", mae)

# sess.close()

