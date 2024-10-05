# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: N.Mahesh
RegisterNumber:  2305001017
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/ex1.csv')
df.head()

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')

from sklearn.model_selection import train_test_split
X = df['X']
y = df['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
X_train_reshaped = X_train.values.reshape(-1, 1)
lr.fit(X_train_reshaped,Y_train)

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train, lr.predict(X_train.values.reshape(-1, 1)), color='red')

m=lr.coef_
m

b=lr.intercept_
b

pred=lr.predict(X_test.values.reshape(-1, 1))
pred

X_test

Y_test

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, pred)
print(f"Mean Squared Error (MSE): {mse}")
```

## Output:
![Screenshot 2024-10-05 134744](https://github.com/user-attachments/assets/f72f4980-c5c3-4ddf-8a89-f65c0eab6055)
![Screenshot 2024-10-05 134836](https://github.com/user-attachments/assets/fcd464c0-4196-485e-80bc-fd5faef41b37)





## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
