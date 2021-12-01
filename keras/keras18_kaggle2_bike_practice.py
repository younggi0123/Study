import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import validation
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error    # r2, mse


def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

# 1. 데이터
path = "./_data/titanic/"
path = "./_data/bike/"
train = pd.read_csv(path + 'train.csv')
# print(train)        # (10866, 12)
test_file = pd.read_csv(path + 'test.csv')
# print(test_file)         # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
# print(submit_file)       # (6493, 2)
# print(submit_file.columns)   #['datetime','count']

x = train.drop( ['datetime', 'casual', 'registered', 'count'], axis=1)
test_file = test_file.drop( ['datetime'], axis=1)

y = train['count']

# 로그변환
y = np.log1p(y)

# Train&test&val_set
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.3)


train = pd.get_dummies(train, columns=['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'])
test_file = pd.get_dummies(test_file, columns=['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'])

# train, test_file = train.align(test_file, join='left', axis=1)
# test_df = test_file.drop(['count'], axis=1)

#그리고 다시 train을 set해본다.
# x_train, x_test, y_train, y_test = train_test_split(train.drop(['count'], axis=1), train['count'], test_size=0.3)

# 2. 모델링 구성
model = Sequential()
model.add(Dense(6, input_dim=8))
model.add(Dense(8, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2스코어 : ', r2)

# RMSLE
rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)

############################################제출용 제작############################################
results = model.predict(test_file)
submit_file['count'] = results
# print(submit_file[:10])
submit_file.to_csv(path + "bike_submit_ver.csv", index=False)



