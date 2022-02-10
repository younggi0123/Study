import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable( [2], dtype=tf.float32 )


# init = tf.global_variables_initializer()
init = tf.compat.v1.global_variables_initializer()

                    # 텐서플로우의 모든 변수는 반드시 초기화 해줘야 함
sess.run(init)      # 이니셜 라이저 위에 선언했던 모든 변수를 초기화하겠단 말(즉, 이 변수를 쓸 수 있어요!란 뜻)\
                    # 예로, x y z A B Г ..... 등을 하나하나 run해 줄순 없을테니까 global로 선언하는 거임 (실제 x외ㅏ init은 메모리 위치가 달라)



print( sess.run(x) )

print( "잘나오니? ", sess.run(x) )







# 지금까지 tf01~06 constant variance placehold 복습할 것