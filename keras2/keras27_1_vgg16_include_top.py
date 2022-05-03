import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

model = VGG16()
# model = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))     # Fullyconnected True
# model.summary()
# ============================include_top=True==================================
# FC Layer 원래 것 그대로 쓰기
#  input_2 (InputLayer)        [(None, 224, 224, 3)]     0
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
# ...
#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________



model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))    # Fullyconnected False
model.summary()
# ============================include_top=False==================================
# FC Layer 원래 것 삭제 ! => 앞으로 커스터 마이징을 할 것이다!
# 예를 들면 input_shape=(32,32,3)으로 바꿀 수 있다.
#  input_2 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
#  ...
#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________



print(len(model.weights))
print(len(model.trainable_weights))



# imagenet은 기본으로 224 224 3 을 쓰는데 넌 32 32 3을 보냈구나 ?
# 그럼 include_top을 false로 고치면 커스터마이징이 되어 inputshape를 조절 가능하게 되는 것이다.







# Fully Connected Layer 정리