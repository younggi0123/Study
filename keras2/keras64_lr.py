# 경사하강법
# 알아서 튜닝 ㄱ


x = 0.5     # x값은 임의로 바꾼다.
y = 0.8     # 목표값
w = 0.5     # 가중치 초기값
lr = 0.1
epochs = 300

for i in range(epochs):
    predict = x * w
    loss = (predict - y) ** 2

    # 가중치와 epoch 도 넣어서 아래 print를 수정하시오
    print("Epoch : ", i+1, "\tweight : ", round(w, 2) ,  "\tLoss : ", round(loss , 4), "\tPredict : ", round(predict, 4) )

    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x * (w + lr)
    down_loss = (y - down_predict) ** 2
    
    if( up_loss > down_loss ):
        w = w - lr
    else:
        w = w + lr




# 예를들어 predict가 10 11을 왔다갔다? lr 이넘크니까 줄여봐

