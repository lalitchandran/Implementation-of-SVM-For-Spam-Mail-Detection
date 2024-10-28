# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S LALIT CHANDRAN
RegisterNumber: 212223240077
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Result output:

![image](https://github.com/user-attachments/assets/7bdc4506-f800-4f15-aded-4a60b0cb7faf)

data.head():

![image](https://github.com/user-attachments/assets/f69b6825-d8ee-40bc-918e-74b78ef0c516)

data.info():

![image](https://github.com/user-attachments/assets/93b25a76-4bec-4740-88cc-d32cfc77505a)

data.isnull().sum():

![image](https://github.com/user-attachments/assets/0efb6c30-ec82-4d8a-9bf9-52359dd28e93)

Y_prediction value:

![image](https://github.com/user-attachments/assets/84eb6cc4-4c47-4932-9e77-7af9168f7aaf)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
