import numpy as np   
import tensorflow as tf 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

tf.compat.v1.set_random_seed(44)
datasets = load_iris()

x_data = datasets.data
y_data = datasets.target
# print(x_data.shape, y_data.shape) #(150, 4) (150, 1)          # 4
y_data = y_data.reshape(-1,1)

print(y_data)   # 000011112222 도배된상태
print(np.unique(y_data)) #[0 1 2]                               # 3

# 참고 : from tensorflow.keras.utils import to_categorical #one hot 라이브러리
# y_data = to_categorical(y_data) #위에서 벡터로 바꿔주는 과정을 얘가 처리해줌 (안하면 기억나쥐? 2가 1의 2배가 아니라던 예시)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 희소행렬을 OneHot 인코딩을 시키기 위해!  OneHotEncoder(categorical_features= [0], sparse=False) 혹은 .toarray()
encoder = OneHotEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()
# scikit-learn의 OneHotEncoder는 그냥 변수에 객체를 다시 할당해주기만 한다.
# toarray()를 쓰면 array형식의 output을 얻을 수 있는데, dataframe으로 다시 매핑할 수는 없다(기존 dataframe의 shape과 안맞으므로)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7, shuffle = True, random_state=42)

#스케일러를 했을때 값이 더 좋아짐 
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

#scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 공간창출
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])


w = tf.Variable(tf.random.normal([4,3]), name = 'weight') 
# (x열 , y열) -> x데이터와 (N, 4) * (4 , 3) -> (None ,3 )이 되기때문에

b = tf.Variable(tf.random.normal([1,3]), name = 'bias') 
#y의 출력 갯수만큼 출력 (행렬의 덧셈방법-행과 열의 갯수가 같아야함)
#더해지는건 1나인데 나가는게(y갯수가) 3개라서 1,3

#소프트맥스
hypothesis = tf.nn.softmax( tf.matmul(x, w) + b )

#카테고리칼 크로스 엔트로피 
loss = tf.reduce_mean( -tf.reduce_sum(y * tf.log(hypothesis), axis=1) )
# categorical_crossentropy


#이전 버전 
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#합친 버전 
# train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

sess = tf.Session() 
sess.run(tf.global_variables_initializer())

for epochs in range(5000):
    loss_val, _ = sess.run([loss, train],
            feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0 :
        print(epochs, "loss : ", loss_val)



# #평가 예측
y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
# [0.123, 0.8323, 0.0124] 이런식으로 값이 나오게 된다 여기서 argmax를 하게 되면 1 이런식으로 라벨값이
# 도출되기때문에 accuracy를 구하려면 y_test를 argmax해주던가 아예 원핫인코딩 단에서 부터 y_test는 원핫인코딩
# 하지 않는 방법을 하면 된다 현재 Y_test의 상태는 [0, 1, 0] 의 상태이기때문에 argmax를 해준다
y_pred = np.argmax(y_pred, axis= 1)
y_test = np.argmax(y_test, axis= 1)

print(y_pred)
print(y_test)

accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), dtype=tf.float32))
a = sess.run([accuracy])

print('accuracy : ',a)

# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0 0 0 2 1 1 0 0]
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0 0 0 2 1 1 0 0]
# accuracy :  [1.0]

sess.close()