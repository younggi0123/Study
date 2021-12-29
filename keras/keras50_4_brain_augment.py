import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=False,
    vertical_flip=False,
    zoom_range=0.2,
    fill_mode='nearest'
)

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train',
    target_size=(100, 100),
    batch_size=160,
    class_mode='binary',
    shuffle=True
)#Found 160 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size=(100, 100),
    batch_size=160,
    class_mode='binary',
    shuffle=True
)#Found 120 images belonging to 2 classes.

# train, test 설정
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

argument_size=int(160*0.2)#32
#