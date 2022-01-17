# from re import M
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype

import warnings#예외처리를 통해 에러메세지를 지우려 씀
warnings.filterwarnings('ignore')

datasets = fetch_covtype()
x= datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# 리스트의 딕셔너리 키 밸류 형태 키=이름, 밸류=값

print("allAlgorithms : ", allAlgorithms)
print( "모델의 개수 : ", len(allAlgorithms) )   # 41

for(name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, "의 정답률 : ", acc) # 이름 출력
        
    except:
        # continue
        print(name, '은 에러난 놈!!!')


# 오래된 algorithm or version 으로 인한 문제로 결과값이 안나오는 경우가 있음