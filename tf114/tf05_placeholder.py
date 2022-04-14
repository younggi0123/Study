
import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a+b

print(sess.run(adder_node))

# 메모리위치 잡아놓고 넣고싶은거 넣겠다
# sess run 하는 순간에 placeholder에 넣고 싶을때 값을 넣어줌(run하는 때에)
# placeholder는 feed_dict와 짝이다 !

print(sess.run(adder_node, feed_dict={ a:3, b:4.5 } ))
print(sess.run(adder_node, feed_dict={ a:[1, 3], b:[3,4] } ))

add_and_triple = adder_node*3
print( sess.run(add_and_triple, feed_dict={a:4, b:2} ))
