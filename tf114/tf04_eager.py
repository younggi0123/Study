import tensorflow as tf
print(tf.__version__)

print(tf.executing_eagerly())   # False

# 즉시실행 True 상태로 켜주면(텐서2가 즉시실행임) 끄면 False임(텐서1 사용가능)

# 즉시실행모드!
tf.compat.v1.disable_eager_execution()      # disable 끄기
# 가상환경 tf270cpu(텐서2)에서 켜보니 돌아감을 알 수 있다.

print(tf.executing_eagerly())   # False

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))

