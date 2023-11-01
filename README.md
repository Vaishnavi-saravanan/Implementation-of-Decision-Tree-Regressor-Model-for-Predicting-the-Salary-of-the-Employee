# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
# AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

# Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
# Algorithm :

1 . Import dataset and get data info
2 . Check for null values
3 . Map values for position column
4 . Split the dataset into train and test set
5 . Import decision tree regressor and fit it for data
6 . Calculate MSE,R2 and y predict.

# Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VAISHNAVI S
RegisterNumber:  212222230165
```
```

import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l0=LabelEncoder()

data["Position"]=l0.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
# Output:
# data.head()
![Screenshot 2023-10-21 222315](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541897/bd6cf209-b8d7-453a-a8cd-d4b4829ac568)

# data.info()
![Screenshot 2023-10-21 222321](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541897/c7f7b4d0-9962-4307-b181-6b50a6a845ca)


# isnull() and sum()

![Screenshot 2023-10-21 222328](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541897/cc9fc2d4-e77c-49b0-a64b-4f0e298b841b)

# data.head() for salary
![Screenshot 2023-10-21 222333](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541897/ef410f5b-d4be-43aa-a65e-5c0e2083eb03)


# MSE Value

![Screenshot 2023-10-21 222336](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541897/7383dc75-4323-4d6d-9415-cd188348e420)

# r2 value

![Screenshot 2023-10-21 222340](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541897/836935da-d917-47de-bc91-b76b59ba2ce1)

# data prediction
![Screenshot 2023-10-21 222346](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541897/4ce1880f-7d97-4315-9e72-f0df408de0e5)


# Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
