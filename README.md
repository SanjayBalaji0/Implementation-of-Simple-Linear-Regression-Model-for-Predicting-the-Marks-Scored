# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.Sanjay Balaji
RegisterNumber:  212223240149
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/student_scores.csv")
df.head()
```
```
df.tail()
```
```
#segregating data to variables
X=df.iloc[:,:-1].values
X
```
```
Y=df.iloc[:,1].values
Y
```
```
#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
```
```
#displaying predicted values
Y_pred
```
```
Y_test
```
```
#graph plot for training data
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours VS Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title("Hours VS Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse=mean_squared_error(Y_test,Y_pred)
print('MSE =',mse)
```
```
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE =',mae)
```
```
rmse=np.sqrt(mse)
print('RMSE =',rmse)
```

## Output:
#  head()
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/4d48f7f4-fe37-4818-acef-f7147b1045ea)
# tail()
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/7c5a3535-65a7-47f3-ace6-fd3ac96b6a0b)
# X
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/90fb4d86-6d3d-45b7-afc2-50c9a9a2acfb)
# Y
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/faab65f7-d043-4402-8d15-8935f47d4ea3)
# Y_pred
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/a58fdd3b-6a3b-46d7-998a-db6eb4d6870b)
# Y_test
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/e0e4a53c-6505-4270-82be-2470abd4a83d)
# Graph 1
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/cf4fbaa2-ecf9-435c-8f7c-5dbaa93296b6)
# Graph 2
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/8cd85143-305d-4051-81e1-b4ff86b1d175)
# MSE
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/b9cd9efe-b5f0-4e1b-88a5-e5498753db39)
# MAE
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/d8fddf05-8be3-4826-ba1a-bf29c81953c6)
# RMSE
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/c0bf3044-eb15-4e32-8bb5-2e3f491a9286)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
