from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

model = VGG16()
model = ResNet50()
model = ResNet50V2()
model = ResNet101()
model = ResNet101V2()
model = ResNet152()
model = ResNet152V2()
model = DenseNet121()
model = DenseNet169()
model = InceptionV3()
model = InceptionResNetV2()
model = MobileNet()
model = MobileNetV2()
model = MobileNetV3Small()
model = MobileNetV3Large()
model = NASNetLarge()
model = NASNetMobile()
model = EfficientNetB0()
model = EfficientNetB1()
model = EfficientNetB7()
model = Xception()
model.trainable = False
model.summary()

print("=================================================")
# print("모델명 : ", ???)
print("전체 가중치 개수 : ", len(model.weights))
print("훈련 가능 가중치 개수 : ", len(model.trainable_weights))

# 모델명 : Xception




