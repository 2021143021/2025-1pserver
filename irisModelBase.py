# 2025_03_10
#프로젝트 2 붓꽃분류기 만들기
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
iris_df = pd.read_csv('iris.csv')

#print(iris_df['sepal_length'].mean())
Y=iris_df['species']
X=iris_df.drop('species', axis=1)
#print(X)
#print(Y)

kn =KNeighborsClassifier()
model_kn = kn.fit(X,Y)

X_new = np.array([[3,3,3,3]])
#X_new = np.array([[5.0,3.4,1.4,0.2]])
#['setosa']
#[[1. 0. 0.]]
prediction = model_kn.predict(X_new)
print(prediction)
probability = model_kn.predict_proba(X_new)
print(probability)