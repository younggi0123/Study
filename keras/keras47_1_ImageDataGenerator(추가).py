# ☆★☆★☆★☆★☆★☆★☆★☆★☆★이미지 데이터를 증폭하여 쉽게 생성하는 방법☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★
# 변조된 데이터를 활용하여 쉽게 print문으로 찍어보면서 데이터를 dir에 생성해준다
# ☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# IDG를 정의한다.
train_datagen = ImageDataGenerator(
    rescale = 1./255,               # 데이터 픽셀 범위 0과 1사이로 scaling하기 위한 인자
    horizontal_flip = True,         # 상하반전(mnist 데이터 등에서 숫자예측시 6과 9는 다른 숫자가 되므로 유의)
    vertical_flip =True,            # 좌우반전
    width_shift_range = 0.1,        # 좌우이동
    height_shift_range = 0.1,       # 상하이동
    rotation_range = 5,             # 회전이동
    zoom_range = 1.2,               # zoom 증폭
    shear_range = 0.7,              # 부동소수점. 층밀리기의 강도입니다. (도 단위의 반시계 방향 층밀리기 각도)
    fill_mode = 'nearest'           # 이동시켜 생긴 빈자리를 근처값과 비슷하게 넣겠다
                                    # 마지막 쉼표는 해도 되고 안 해도 되는데 안하는게 많다.
                                    # (100장으로 너무 적어서 9900장을 증폭시켜서 10000장을 돌렸다.)
                                    # 선택적으로 쓸거 쓴다.
)

# train과 test를 동일하게 하지 않은 이유는?
# = 평가할 때는 대상 사진에 대해서 평가만 하면 되는거라 변조가 필요 없다.
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    # 사용 예시
    # (directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest') -> DirectoryIterator
    
    # 경로 설정 => train과 test의 경로상 디렉토리(폴더)를 동일하게 일치시킨다.
    '../_data/image/brain/train/',
    target_size = (150, 150),       # 이미지들을 수작업으로 다 사이즈 조정해 줄 필요 없이 알아서 다 100, 100으로 조정해 준다.
    batch_size = 5,                 # y값이 5개가 나와
    class_mode = 'binary',
    shuffle = True, seed=66, # just like random_state
    
    
    # ☆★☆★☆★☆★☆★☆★☆★
    color_mode='grayscale',
    save_to_dir='../_temp/'
    # ☆★☆★☆★☆★☆★☆★☆★
    
    
    
    
)   # 출력 : Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary',
    shuffle = False # just like random_state # test는 평가만 하면 되기에 굳이 suffle 안해도 된다.(안 써도 됨)
)   # 출력 : Found 120 images belonging to 2 classes.

print(xy_train) # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000024A3DE14F70>
# 데이터에 대한 직접 접근 아님

print( xy_train[31] )
# 데이터를 건드렸기에 31번째를 출력하면서 save_to_dir에 같이 생성됨
# ☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★

print(xy_train[0][0])
print(xy_train[0][1])
# print(xy_train[0][2])      # IndexError: tuple index out of range
print(xy_train[0][0].shape, xy_train[0][1].shape)     # (5, 150, 150, 3) (5,)   # 라벨이 두 개니가 (5, 2)도 가능(원핫, 카테고리컬크로스엔트로피, 소프트맥스)
#정의한걸 flowfrom으로 실제 데이터로 땡겨오고 사이즈 맞춰주며 ~~(정리 ㄱㄱ)

# ☆★찍은것들이 다 dir에 저장되었다.


'''

# 데이터 구조 파악
print(type(xy_train))       # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>   0번째는 튜플이였다.
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>
# 위를 통한 모델링 구성

'''