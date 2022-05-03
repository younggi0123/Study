
import numpy as np

x = np.array( [
                [1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99],
                                
                ] ) # 2행 13열

# (2, 13) => (13, 2)
x = np.transpose(x)
# print(aaa)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)

### -1을 "Nan"으로 바꿔서
### interpolate를 하기위한 시도..

for i in range(12):
    col = x[:,i].reshape(-1,1)
    outliers.fit(col)
    Ol = outliers.predict(col)
    
    Pl=np.where(Ol==-1,"Nan",Ol)



    # Ol 안에서 bool자든 뭘 이용해서 어떤조건이 == -1 이면 그 자리의 -1에 평균값 

