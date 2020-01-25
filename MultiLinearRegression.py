#--------------------------------------Multilinear Regression------------------------------------

# =============================================================================
# Importing the libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
%matplotlib qt

# =============================================================================
# Importing the dataset
# =============================================================================
dataset = pd.read_csv('C:\\Users\\Administrator\\Desktop\\Data Science\\Machine Learning A-Z New\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# =============================================================================
# Encoding the categorical data
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# =============================================================================
# Avoiding the dummy variable trap
# =============================================================================
x = x[:, 1:]

# =============================================================================
# Splitting the dataset into training set and test set
# =============================================================================
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 0)

# =============================================================================
# Fitting the multiple linear regression to the training set
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# =============================================================================
# Predicting the Test set result
# =============================================================================
y_pred = regressor.predict(x_test)

# =============================================================================
#Bbuilding the optimal model using the backword eliminaition
# =============================================================================
x = np.append(arr = np.ones((50,1)).astype(int),values = x, axis = 1)
x_opt = x[:, [0,1,2,3,4,5]]
model = sm.OLS(endog = y, exog = x_opt).fit()
model.summary() 

x_opt = x[:, [0,1,3,4,5]]
model = sm.OLS(endog = y, exog = x_opt).fit()
model.summary()


x_opt = x[:, [0,3,4,5]]
model = sm.OLS(endog = y, exog = x_opt).fit()
model.summary()


x_opt = x[:, [0,3,5]]
model = sm.OLS(endog = y, exog = x_opt).fit()
model.summary()


x_opt = x[:, [0,3]]
model = sm.OLS(endog = y, exog = x_opt).fit()
model.summary()

######OR######

# Backward elimination
x_train_0 = sm.add_constant(x_train)
x_test_0 = sm.add_constant(x_test)
ols = sm.OLS(endog=y_train, exog= x_train_0).fit()
print(ols.summary())

#################################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

%matplotlib qt

dataset = pd.read_excel('insurance.xlsx')

#assign dummy variable to the categorical features(sex)

#Encoding the features
from sklearn.preprocessing import LabelEncoder

#smoker
labelencoder_smoker = LabelEncoder()
dataset.smoker = labelencoder_smoker.fit_transform(dataset.smoker)
#sex
labelencoder_sex = LabelEncoder()
dataset.sex = labelencoder_sex.fit_transform(dataset.sex)
#region
labelencoder_region = LabelEncoder()
dataset.region = labelencoder_region.fit_transform(dataset.region)

from sklearn.model_selection import train_test_split

x = dataset.iloc[:,:6].values
y = dataset.iloc[:,6].values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


x = np.append(arr = np.ones((1338,1)).astype(int),values = x, axis = 1)
x_opt = x[:, [0,1,2,3,4,5,6]]
model = sm.OLS(endog = y, exog = x_opt).fit()
model.summary()

x_opt = x[:, [0,1,3,4,5,6]]
model = sm.OLS(endog = y, exog = x_opt).fit()
model.summary()