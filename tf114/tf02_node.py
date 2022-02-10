import tensorflow as tf

# 덧셈 연산

# 텐서연산을 위해 텐서플로란걸 명시해 줘야함
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)

print(node3)
# 출력결과 Tensor("add:0", shape=(), dtype=float32) # 자료형만 던져줌

#      ▼

# 세션만들어야!


# ★★★★★★★★★★★★ Session 텐서머신 구동 ★★★★★★★★★★★★★★
# sess = tf.Session()
# 위와 같이구동하자 아래와 같은 오류가떴다/

# 얘를 적시하거나
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# WARNING:tensorflow:From d:\Study\tf114\tf03_node2.py:8: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
# 오류 => pip 오류가 있어서 CMD에서 넘파이를 제거했다가 다시깔아서 버전을 바꿔줌 1.16.6으로.
sess = tf.compat.v1.Session()           # tensorflow 폴더구조에서 세션을 바로가져왔다면, 
                                        # 1.13부터 텐서플로안에 compat이란 폴더구조가 생기고
                                        # 그안에 v1그안에 session이있어 여기서 가져오겠다. 를 칭한 것




# print('node1, node2 : ', sess.run(node1, node2))       # Error! 두개 이상 출력하려면 리스트써야겠죠?
print('node1, node2 : ', sess.run([node1, node2]))       # 두개 이상 출력하려면 리스트써야지

print('node3 : ', sess.run(node3))
# 출력결과 : 7.0
