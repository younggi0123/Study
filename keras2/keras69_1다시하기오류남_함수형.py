# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Flatten, Input
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.datasets import cifar100
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# # [실습] 아래를 수정


# scaler = StandardScaler()

# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# n = x_train.shape[0]# 이미지개수 50000
# x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
# x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1

# x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

# m = x_test.shape[0]
# x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)
# # print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 100)
# # print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 100)


# input = Input(shape=(32,32,3))
# vgg16 = VGG16(include_top=False)(input)
# hidden1 = Dense(100)(vgg16)
# # hidden2 = Dense(Flatten)
# hidden2 = Dense(80)(hidden1)
# hidden3 = Dense(64)(hidden2)
# hidden4 = Dense(32)(hidden3)
# output1 = Dense(10)(hidden4)
# model = Model(inputs=input, outputs=output1)

# # model.summary()

# # 3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )               # loss는 categorical_crossentropy가 된다.+ 모든분류에서 accuracy가 가능하다(보조지표 metrics)

# # Fit
# # from tensorflow.keras.callbacks import EarlyStopping
# # es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
# #                    verbose=1, restore_best_weights=False)    

# model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1,    #표본이 50만개인데 설마 배치사이즈1을 하진 않겠죠???
#           validation_split=0.3)  #, callbacks=[es])

# # 4. 평가, 예측
# # Evaluate
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
# print('accuracy : ', loss[1]) 



# git으로 다른 친구것
from pickletools import optimize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, GlobalAvgPool2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


input = Input(shape=(32, 32, 3))
vgg16 = VGG16(include_top=False)(input)
vgg16.trainable = True

gap = GlobalAvgPool2D()(input)

hidden1 = Dense(128, activation='relu')(gap)
dropout1= Dropout(0.2)(hidden1)
hidden2 = Dense(32, activation='relu')(dropout1)
dropout2 = Dropout(0.2)(hidden2)
output = Dense(10, activation='softmax')(hidden1)

model = Model(inputs=input, outputs=output)

model.summary()

optimizer = Adam(learning_rate=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='amin', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
print('loss, acc ', loss)


'''
걸린 시간 :  439.44
loss, acc  [1.9946000576019287, 0.2581000030040741]
'''