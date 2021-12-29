# Fashion_Mnist 수치확인용(수치화 된 데이터)
# ☆★ 참 고 ★☆
# ☆★https://circle-square.tistory.com/108★☆
# ☆★https://circle-square.tistory.com/108★☆
# ☆★https://circle-square.tistory.com/108★☆


from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image as keras_image
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# IDG를 정의한다.
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip =True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 5,
    zoom_range = 1.2,
    shear_range = 0.7,
    fill_mode = 'nearest'
)
augment_size = 100# 100개만큼 반복(증폭. 하겠다.),  

# 밑에서 사용할 애들이 어떻게 생긴지 미리 확인해 봄.
# 784로 reshape
print(x_train[0].shape)                 # (28, 28)
print(x_train[0].reshape(28*28).shape)  # (784, )
print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1).shape)  # (100, 28, 28, 1)


x_data = train_datagen.flow(
    # numpy.tile 함수는 어레이를 (타일과 같이) 지정한 횟수(argument)만큼 반복합니다.
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),       # x
    
    #0으로 채워넣을거얌
    np.zeros(augment_size),                                                        # y
    
    batch_size = augment_size, 
    shuffle = False
).next()                                                                            # 100개가 도는 것
# .next를 빼고 돌리면 x_data[0].shape 부분에서 오류가 발생하는데 왜 발생하겠는가??
# next() 함수는  - 반복 가능 객체의 다음 요소 반환이다.★
# → 찾아볼 것


print(type(x_data)) # <class 'tuple'>   # → IDG할때 데이터 변환할때 0번째가 튜플 상태였다. 00 이 넘파이 상태 01 이 y  01의 앞의 0은 batch였다(재확인)
# print(x_data)
print(x_data[0].shape, x_data[1].shape)     # (100, 28, 28, 1) (100, )

# augment의 개수 중 49개만 출력해 본다.(but, augment가 49보다 작을 경우 for문을 못 도니까 오류 뜸)
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()















# # 원래 이미지에 증식된 이미지를 추가
# x_train = np.concatenate((x_train, x_augment))
# y_train = np.concatenate((y_train, y_augment))
# print(x_train.shape) # (90000, 28, 28, 1)