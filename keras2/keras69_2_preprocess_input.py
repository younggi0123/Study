from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


#디텍션 쓸 것도 없이 전이학습 모델로 훈련해서 해결하면 이미지 대회도 장땡
model = ResNet50(weights='imagenet')

sample_directory = '../_data/image/_predict/cat_dog/'
sample_image = sample_directory + "cat_dog.jpg"
# img_path = '../_data/image/_predict/cat_dog/cat_dog.jpg'

img = image.load_img(sample_image, target_size=(224, 224))
x = image.img_to_array(img)          # 이미지 수치화
print("=========================================image.img_to_array(img)=========================================")
print(x, '\n', x.shape)         # (224, 224, 3)

x = np.expand_dims(x, axis=0)
print("=========================================np.expand_dims(x, axis=0)=========================================")
# print(x, '\n', x.shape)         # (1, 224, 224, 3)

# ResNet모델에 imagenet 데이터를 썼을때 최적의 preprocessing이다.
x = preprocess_input(x)
print("=========================================preprocess_input=========================================")
# print(x, '\n', x.shape)         # (1, 224, 224, 3)

preds = model.predict(x)
print(preds, '\n', preds.shape)

print('결과는 : ', decode_predictions(preds, top=5)[0]) # just like argmax




# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# img = image.load_img(sample_image, target_size=(224, 224))

# x = preprocess_input(x)

# preds = model.predict(x)
# print(preds, '\n', preds.shape)
# print('결과는 : ', decode_predictions(preds, top=5)[0]) # just like argmax