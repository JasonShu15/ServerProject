#coding:utf-8
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
dataSet=pd.read_csv("D://Personal//Desktop//sunite.csv")
dataSet_x=dataSet['rmse'].tolist()
dataSet_y=dataSet['accuracy'].tolist()
X=np.array(dataSet_x).reshape(-1,1)#type:numpy.ndarray
Y=np.array(dataSet_y).reshape(-1,1)
model=LinearRegression()
model.fit(X,Y)
intercept=model.intercept_#截距
coef=model.coef_#系数,斜率
D_value=dataSet_y-coef*dataSet_x-intercept#求差值
listValue=D_value[0].tolist()
for i in range(5):
    maxIndex=listValue.index(max(listValue))
    restValue=listValue.pop(maxIndex)#找到最大的五个差值
for i in range(len(listValue)):
    if listValue[i]==restValue:
        max_x=dataSet_x[i]
        print max_x
























#import matplotlib.pyplot as plt
#plt.scatter(X,Y,color='black',marker='o')
#plt.scatter(X,model.predict(X),color='blue',marker='+')
#plt.xlabel('rmse')
#plt.ylabel('accuracy')
#plt.show()