# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Use the standard libraries in python for Gradient Design.
2.Upload the dataset and check any null value using .isnull() function.
3.Declare the default values for linear regression.
4.Calculate the loss usinng Mean Square Error.
5.Predict the value of y.
6.Plot the graph respect to hours and scores using scatter plot function

## Program:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv('student_scores - student_scores.csv')
df.head()
df.tail()

#checking for null values in dataset
df.isnull().sum()

#To calculate Gradient decent and Linear Descent
x=df.Hours
x.head()

y=df.Scores
y.head()

n=len(x)
m=0
c=0
L=0.001
loss=[]
for i in range(10000):
    ypred = m*x + c
    MSE = (1/n) * sum((ypred - y)*2)
    dm = (2/n) * sum(x*(ypred-y))
    dc = (2/n) * sum(ypred-y)
    c = c-L*dc
    m = m-L*dm
    loss.append(MSE)
print(m,c)

#plotting Linear Regression graph
print("Slope = {}\nConstant = {}".format(m,c))
y_pred=m*x+c
plt.scatter(x,y,color="magenta")
plt.plot(x,y_pred,color="red")
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study hours vs Scores")
plt.show()

#plotting Gradient Descent graph
plt.plot(loss, color="darkblue")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
/*
Program to implement the linear regression using gradient descent.
Developed by:T.vishal 
RegisterNumber:212223100060  
*/
```

## Output:
![Screenshot 2024-03-06 111043](https://github.com/VISHAL123456789V/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161364099/3a9fd772-3a44-4c8e-9d06-534b2768cab7)
![Screenshot 2024-03-06 112100](https://github.com/VISHAL123456789V/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161364099/bf33ce11-a4d4-450a-8df6-7d2fe8ca4007)







## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
