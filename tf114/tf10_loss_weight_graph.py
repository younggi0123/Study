import tensorflow as tf
import matplotlib.pyplot as plt

# 한글 폰트 사용을 위해서 세팅
# https://wikidocs.net/45798
# from matplotlib import font_manager, rc
# import platform
plt.rc('font', family='Malgun Gothic') # 폰트 저작권 주의 !
plt.rcParams['axes.unicode_minus'] = False # 한글 폰트 사용시 마이너스 폰트 깨짐 해결 (중요!)(중요!)(중요!)(중요!)(중요!)(중요!)(중요!)(중요!)
#matplotlib 패키지 한글 깨짐 처리 끝



x = [1,2,3]
y = [1,2,3]
w = tf.placeholder(tf.float32)

hypothesis = x * w
loss = tf.reduce_mean(tf.square(hypothesis-y))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={ w:curr_w })

        w_history.append(curr_w)
        loss_history.append(curr_loss)

print("================================ W history ===============================")
print(w_history)
print("============================== loss history ==============================")
print(loss_history)
print("==========================================================================")
# w위치별 loss값 보여주는 것임 # w 가 epoch에 따라 급감하기에 이는 보여주기용임

plt.plot(w_history, loss_history)
plt.xlabel("weight")
plt.ylabel("loss")
plt.title(" 한글깨짐발생 ")
plt.show()