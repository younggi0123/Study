import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv


#1. 데이터 
path = '../_data/kaggle/bike/'   
train = read_csv(path+'train.csv')  
print(train.shape)      # (10886, 12)
test_file = read_csv(path+'test.csv')
print(test_file.shape)    # (6493, 9)
# submit_file = read_csv(path+ 'sampleSubmission.csv')
# print(submit_file.shape)     # (6493, 2)
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']



from sklearn.model_selection import train_test_split, KFold, cross_val_score
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
model = SVC()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x)
x_test = scaler.transform(x)


scores = cross_val_score(model, x, y, cv=kfold)
print(" R2 : ", scores, "\n Cross_Val_Score : ", round(np.mean(scores), 4) )


#  ACC :  [0.01836547 0.0192926  0.01745521 0.01240239 0.01423978] 
#  Cross_Val_Score :  0.0164
