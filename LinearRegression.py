#----------------------------------------Linear Regression---------------------------------------

# =============================================================================
# Importing libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib qt
# =============================================================================
# Importing the dataset
# =============================================================================
dataset =pd.read_csv('C:\\Users\\Administrator\\Desktop\\Data Science\\R\\Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# =============================================================================
# Splitting the dataset into trainingset and test set
# =============================================================================
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state = 0)

# =============================================================================
# Fitting simple linear regression to the Trainign set
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# =============================================================================
# Predicting the Test set result
# =============================================================================
y_pred = regressor.predict(x_test)

# =============================================================================
# Visualising the Training set result
# =============================================================================
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
