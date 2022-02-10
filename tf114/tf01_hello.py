import tensorflow as tf
print(tf.__version__)
# 텐서 1 # 1.14.0

# tf.constant 상수
# tf.variable 변수
# tf.placeholder 입력시키는 곳에 집어넣는 것

hello = tf.constant("Hello World")
print(hello)
# 출력결과 :  Tensor("Const:0", shape=(), dtype=string)


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





print(sess.run(hello))
# 세션만든 후 출력결과 : b'Hello World' (b는 임의로 만드때 생기는 것)




















# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=complusblog&logNo=221237818389 


# 텐서플로우(TensorFlow) 텐서 기본 개념 - Tensor란 무엇인가? (Rank, Shapes, Types)

# 파이썬 텐서플로우(TensorFlow) 스터디 관련 글 더보기 텐서플로우 예제 깃허브(GitHub) 페이지텐서플...

# blog.naver.com


# 1&2차이

# https://givemethesocks.tistory.com/63

# [tensorflow] keras (생활코딩) - tensorflow 1.x / 2.x 차이

# 0. 들어가기 -. tensorflow내 keras 프레임워크 사용법 익히기 -. 생활코딩 keras 강의 들으면서 정리함. tensorflow 버전에 따른 설명이 좀 부족하긴 하네.. 1. keras 사용법 예제 -. tensorflow 버전에 따라 ker..

# givemethesocks.tistory.com


# constant에 hello하나를 집어넣은것을 정상출력하려면

# 그래프를 만들고(변수를 정의 하고), 노드를 만들어서 출력하면 그냥 데이터 타입만 나오기에

# 텐서머신에 집어넣어야한다. (= sess.run★을 통과시킨다 ) (연산자체가 그래프화된다)

# 쉽게말해 hello world 찍으려면 sess.run을 통과시켜줘야 한다는 뜻.

# 먼저, session을 만들어준다(= 머신을 만들겠다.)



# 노드끼리의 연산

# ex) w= x+b라면 ?

# x    w    b

#   곱    .

#      +

#      ↓

#   sess머신