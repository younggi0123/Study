import tensorflow as tf
tf.compat.v1.set_random_seed(66)

변수 = tf.compat.v1.Variable( tf.random_normal([1]), name='weight' )    # input_dim = 1
print(변수)


# 방법 1.
sess = tf.compat.v1.Session()
sess.run( tf.compat.v1.global_variables_initializer() )
aaa = sess.run(변수)       # tf형을 사람형으로 변환
print("aaa : ", aaa)    # w에 있는 랜덤값이 들어감(고정된 seed값으로)
sess.close()

# aaa :  [0.06524777]



# 방법 2. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)          # 변수.eval
print("bbb : ", bbb)
sess.close()


# 방법 3.
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print("ccc : ", ccc)
sess.close()

# 변수 초기화는 동일,  sess 사용법은 다양함.
# 출력하는 방법이 1. sess.run 말고도
# 2. 변수.eval의 경우에는 앞의 정의가 session으로 정의시 eval안에 session에대한 명시가 들어가야함
# 3. interaction은 sess.run 안 해도 됨