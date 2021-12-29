# ☆★ 참 고 ★☆
# ☆★https://circle-square.tistory.com/108★☆
# 49-1 copy

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image as keras_image
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# 지금까지의 이미지 데이터의 예측 정확도가 떨어졌던건
# 내가 너무 심한 변형을 가했기 때문일 수 있다.
# 이에 변형 정도를 조정해보았다.
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    # vertical_flip =True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    # rotation_range = 5,
    zoom_range = 0.1,   #zoom range 1.1 to 0.1
    # shear_range = 0.7,
    fill_mode = 'nearest'
)   # plot을 통해 결과를 찍어보니 keras49_1_flow의 신발 데이터들 보다 더 보기에 나은 데이터가 되었다.

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)    # randint = random int, randidx = randindex이다.
# └ mnist의 [0]은 60000이고 이 중 size라는 parameter에 argument_size만큼 40000개의 값을 중복포함없이 랜덤하게 뽑아서 변조하겠다.
# total original images 60,000  +  picked transformed images 40,000 = 100,000장으로 증폭
print(x_train.shape[0])                 # 60000
print(randidx)                          # [19388 40444  5836 ... 51885 13813  7103]
print(np.min(randidx), np.max(randidx)) # 1 59998
# 현재 randindex의 형태는 list이다.

x_augmented = x_train[randidx].copy()   # 
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)                # (40000, 28, 28)   # 40000개 들어가고, shape는 28, 28이 들어가겠다.
print(x_augmented.shape)                # (40000, )

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

# 밑에서 x_train도 합칠때 동일한 shape여야 하니까 reshape해야함
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)



# # 주의 ! 아래는 단순한 결합이다. train_datagen을 써서 결합해줘야한다.
# x_train = np.concatenate( (x_train, x_augmented) )  # concatenate는 괄호가 두개이다.  why!?
# print(x_train)
# print(x_train.shape)                    # (100000, 28, 28)  # random한 40000개를 더하여 10만개가 되었다.!


# y_augment인데 np.zeros도 같은거 찾아봐.
# x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size), 
#                                  batch_size = augment_size, shuffle=False,
#                                  ).next()[0]

#                                   =

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size = augment_size, shuffle=False,
                                 ).next()[0]

print(x_augmented)
print(x_augmented.shape)                    # (40000, 28, 28, 1)
# ValueError: ('Input data in `NumpyArrayIterator` should have rank 4. You passed an array with shape', (40000, 28, 28))
# shape 4차원인데 rgb채널이 빠져있다. 28, 28만 받아들이는게 아닌 28, 28, 1을 받아야 한다. flow에 넣을수 있게 x_argment를 reshape해야 하겠다.

# concatenate괄호두개concatenate괄호두개concatenate괄호두개concatenate괄호두개concatenate괄호두개concatenate괄호두개concatenate괄호두개concatenate괄호두개
x_train = np.concatenate((x_train, x_augmented))      #(100000, 28, 28, 1)
y_train = np.concatenate((y_train, y_augmented))

print(x_train)
print(x_train.shape)


'''



# 밑에서 사용할 애들이 어떻게 생긴지 미리 확인해 봄.
# 784로 reshape 함.
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








'''






# # 원래 이미지에 증식된 이미지를 추가
# x_train = np.concatenate((x_train, x_augment))
# y_train = np.concatenate((y_train, y_augment))
# print(x_train.shape) # (90000, 28, 28, 1)