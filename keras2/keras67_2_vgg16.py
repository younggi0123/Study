from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

# vgg16.summary()
vgg16.trainable = False # 가중치를 동결시킨다. !

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))
model.summary()
# 빠르고 vgg16에 대한 최상의 가중치를 가져온다
# 내가만든 훈련모델에서 vgg16을쓸때 이 가중치를 가져다 쓸지 다시 훈련 시킬3지.
# trainable을 true한 상태에섯 데이터를 돌려 다시 훈련시키면 가중치 갱신이 될 것
# 하지만 false로 한다면 1400만개의 가중치를 동결된 계산을 하겠다는 것.
# gpt를 쓰든 뭐를 쓰든 동결할 지 말지는 정해서 하는 것 # 아얘 갱신 못하도록 막아둔 모델도 있긴하다.


                                        # Trainable : True
print(len(model.weights))               # 30

print(len(model.trainable_weights))     # 30