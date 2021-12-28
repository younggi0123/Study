import numpy as np
# 넘파이 로드(save에서 파일명 참고)
# np.save('./_save_npy/keras47_5_train_x.npy', arr=xy_train[0][0])
# np.save('./_save_npy/keras47_5_train_y.npy', arr=xy_train[0][1])
# np.save('./_save_npy/keras47_5_test_x.npy', arr=xy_test[0][0])
# np.save('./_save_npy/keras47_5_test_y.npy', arr=xy_test[0][1])

x_train = np.load('./_save_npy/keras47_5_train_x.npy')
y_train= np.load('./_save_npy/keras47_5_train_y.npy')
x_test = np.load('./_save_npy/keras47_5_test_x.npy')
y_test = np.load('./_save_npy/keras47_5_test_y.npy')

# 2. 모델 구성



# model.evaluate에 batch를 명시하지 않아왔지만 원래 batch_size가 존재했단 소리지.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
model = Sequential()
model.add(Conv2D( 32, (2,2), input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=2, strides=1, padding="VALID"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # bc가 낮은거 metrics높은거 잡아주겠지??????
# model.fit(xy_train[0][0], xy_train[0][1])
hist = model.fit(x_train, y_train, epochs=200, batch_size= 32, validation_split = 0.3)


# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
# 그래프 시각화
# history 시리즈 정리 ㄱㄱ

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

import matplotlib.pyplot as plt
print('loss : ', loss[-1]) # 마지막이 최종 것
print('val_loss : ', val_loss[-1])
print('accuracy : ', acc[-1])
print('val_ accuracy : ', val_acc[-1])

epochs = range(1, len(loss)+1)

# #plot keras13
plt.figure(figsize=(9, 5))
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='loss')
# plt.plot(hist.history['acc'], marker='.', c='green', label='acc')
# plt.plot(hist.history['val_acc'], marker='.', c='yellow', label='val_acc')

plt.plot(epochs, loss, 'r--', label='loss')
plt.plot(epochs, val_loss, 'r:', label="val_loss")
plt.plot(epochs, acc, 'b--', label='acc')
plt.plot(epochs, val_acc, 'b:', label='val_acc')

plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()



'''