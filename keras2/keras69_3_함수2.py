# .output .input .layers



from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = InceptionV3(weights='imagenet', include_top=False)

# base_model.summary()

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

output = Dense(333, activation='softmax')(x)

model = Model(input=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.summary()

print(base_model.layers)#일반모델 레이어 확인