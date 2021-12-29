#훈련데이터를 기존 데이터에서 20% 만큼 증폭 
#성과비교
#save_dir도 temp에 넣은후 삭제할것 

from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train),(x_test,y_test) = mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                    horizontal_flip=True,
                    vertical_flip=False,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    rotation_range=5,
                    zoom_range=0.1,
                    shear_range=0.5,
                    fill_mode='nearest'
                    )


augment_size = 12000 #배치사이즈
# print(x_train.shape) #(60000, 28, 28)
randidx = np.random.randint(x_train.shape[0], size=augment_size)
                              #가져올놈         사이즈만큼 가져올놈에서 데려오겠다 


#print(randidx) #[  293 26736  8034 ...  2473 23220 11920]랜덤하게 들어감 
# print(randidx.shape) #(12000,)배치 사이즈만큼 랜덤하게 들어감 

# #! <x,y 만들기 > - 새로운 12000개를 만들고 기존의 60000개와 합치면 72000개가 된다 

# # 72000개 만들기
x_augmented = x_train[randidx].copy() #메모리가 공유되는걸 방지하기 위해 카피해서 진행.. 
y_augmented = y_train[randidx].copy()
# #같은위치에 있는 12000개가 저장이 된다 x_augumented, y_augumented - 똑같은 이미지가 12000개 생성되어 있음 
# #y값은 똑같은 라벨값이니 y값은 바뀌면 안된다 x의 데이터만 오른쪽 아래 위로 가듯 변화만 준다 
# #12000장의 데이터를 약간씩 수정한다 

# #^^ x_augumented - 12000개 가져오고, y = np.zeros(augment_size)  각각 하나의 데이터가 한장씩 바뀜
# #^^  사만장의 데이터가 사만장으로 바뀜 한장이 다른 한장으로 다른한장이 다른한장으로 바뀜 
# #^^ 이 과정에서 y값은 그대로기 때문에 바뀔 필요강 없다


#print(x_augmented.shape) #(12000, 28, 28)
x_augmented = x_augmented.reshape(x_augmented.shape[0],28,28,1)  #4차원으로 만들어주기 
# #이터러블 넘파이가 4차원을 받기때문에 쉐이프를 바꿔줘야함 
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False,
                                 save_to_dir='../_temp/',
                                 ).next()[0]
                                #x는 [0]에 있고 y는 [1]에 있어서 마지막에 [0]을 붙임으로서 x만 뽑아줌
                                # [0]으로 하면 shape(12000, 28, 28, 1) [1]로 하면 (12000,)

# print(x_augmented.shape)#(12000, 28, 28, 1)
# print(type(x_augmented))
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
#print(x_train.shape) #(12000, 28, 28, 1)



#!######################################################################

#1. 데이터 

# print(x_train.shape, y_train.shape) # (72000, 28, 28, 1) (72000,)- 얘는 4차원을 받는데 3차원이라서 차원을 늘려야한다 
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)



# x_train = x_train.reshape(72000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# from sklearn.preprocessing import StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# # #scaler = MinMaxScaler()
# scaler = StandardScaler()
# # #scaler = MaxAbsScaler()
# # #scaler = RobustScaler()
# # scaler = QuantileTransformer()
# # #scaler = PowerTransformer()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(x_test)

# #차원 늘려주기 
x_train = x_train.reshape(72000, 784,1) 
x_test = x_test.reshape(10000, 784, 1) 



#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(200, 2,input_shape=(784,1)))
model.add(Flatten())
model.add(Dense(150))
model.add(Dense(54))
model.add(Dense(22))
model.add(Dense(10, activation='softmax')) # 왜 시그모이드를 사용할까? 


#sparse_categorical을 통해 원핫 인코딩이 필요 없게된다.
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
                    metrics=['acc'])


hist = model.fit(x_train,y_train, epochs=100,batch_size=500,
                    validation_split=0.2)
                   # validation_steps=4) 
                    #validation_steps=4 이런 파라미터가 있다 


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


loss = model.evaluate(x_test, y_test)
print('loss : ', loss )

#print('acc 전체 : ', acc)

print('acc : ', acc[-1])
print('val acc : ', val_acc[-1])


temp = model.predict(x_test)
#print('원본 : ', temp[:5])
# temp = tf.argmax(temp, axis=1)
# temp = pd.DataFrame(temp)
print('예측값 : ', temp[:5])
print('원래값 : ',y_test[:5])