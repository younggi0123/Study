# https://sdc-james.gitbook.io/onebook/4.-and/5.3.-mnist-dataset/5.4.1.-tensorflow
# https://excelsior-cjh.tistory.com/151  ▶ 여기서 기초보기


# 사칙연산 실습
# 덧셈 node3
# 뺄셈 node4
# 곱셈 node5
# 나눗셈 node6

import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

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


# node3 = node1+node2

add_input = tf.add(node1, node2, name='Add')
sub_input = tf.subtract(node1, node2, name='Sub')
mul_input = tf.multiply(node1, node2, name='Multiply')
div_input = tf.divide(node1, node2, name='Div')
pow_input = tf.pow(add_input, mul_input, name='Power')

with tf.Session() as sess:
    add_output, sub_output, mul_output, div_output = sess.run([add_input, sub_input, mul_input, div_input])
    print("Add=", add_output )
    print("Sub=", sub_output )
    print("Mul=", mul_output )
    print("Div=", div_output )
    
    
# Add= 5.0
# Sub= -1.0
# Mul= 6.0
# Div= 0.6666667

    
    
    
    
    
    
    
    
    
    
    
# 텐서플로에서는 위의 코드처럼 곱셈, 덧셈, 뺄셈을 텐서플로의 tf.<operator>를 사용하여 나타낼 수 있을 뿐만아니라 축약 연산자 즉, *, +, - 등을 사용할 수 있다.

# TensorFlow    연산	축약 연산자	설명
#------------------------------------
# tf.add()	    a + b	a와 b를 더함
# tf.multiply()	a * b	a와 b를 곱함
# tf.subtract()	a - b	a에서 b를 뺌
# tf.divide()	a / b	a를 b로 나눔
# tf.pow()	    a ** b	​a^b를 계산
# tf.mod()	    a % b	a를 b로 나눈 나머지를 구함


# 출처: https://excelsior-cjh.tistory.com/151 [EXCELSIOR]