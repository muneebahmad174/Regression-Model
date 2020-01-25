#-----------------------------Rainfall prediction------------------------------
# =============================================================================
# importing the libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib qt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn import preprocessing

# =============================================================================
# importing the dataset
# =============================================================================
dataset = pd.read_excel('rainfall.xlsx')

# =============================================================================
# checking the missing dataset
# =============================================================================
dataset.isnull().sum()

# =============================================================================
# filling the missing values with the mean
# =============================================================================
dataset = dataset.fillna(np.mean(dataset))

dataset.groupby('SUBDIVISION').size()

x = dataset[['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']]
y = dataset.ANNUAL

# =============================================================================
# splitting the dataset into training and test
# =============================================================================
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3)

# =============================================================================
# importing the linear model and fitting the test set
# =============================================================================
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

# =============================================================================
# predicting the test set
# =============================================================================
y_pred = regressor.predict(x_test)

# =============================================================================
# calculating the accuracy of the model
# =============================================================================
r2_score(y_test,y_pred)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

mean_squared_error(y_test,y_pred)

mean_absolute_error(y_test,y_pred)

