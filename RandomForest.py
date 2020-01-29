#------------------------------------------Random Forest (case 1)--------------
# =============================================================================
# Importing the libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib qt
# =============================================================================
# Importing the dataset
# =============================================================================
dataset = pd.read_csv('C:\\Users\\Administrator\\Desktop\\Data Science\\R\\Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# =============================================================================
# Fitting the random forest regression to the dataset
# =============================================================================
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

# =============================================================================
# predicting the values
# =============================================================================
y_pred = regressor.predict(6.5)

# =============================================================================
# Visualising the regressor model using the High Resolution And Smoother Curve
# =============================================================================
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('salary vs experience')
plt.xlabel('years')
plt.ylabel('salary')
plt.show()
#####################################Decision Tree(case 2)#####################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

%matplotlib qt

dataset = pd.read_excel('insurance.xlsx')


dataset.drop(["region"], axis=1, inplace=True)

x = dataset.drop(["charges"], axis=1)
y = dataset.charges.values  

from sklearn.preprocessing import LabelEncoder

labelencoder_sex = LabelEncoder()
x.sex = labelencoder_sex.fit_transform(x.sex)

labelencoder_smoker = LabelEncoder()
x.smoker = labelencoder_smoker.fit_transform(x.smoker)

x["bmi"] = (x - np.min(x))/(np.max(x) - np.min(x)).values

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor  # Import Decision Tree Regression model

decision_tree_reg = DecisionTreeRegressor(max_depth=5, random_state=13)

decision_tree_reg.fit(x_train, y_train)  # Fit data to the model
from sklearn.model_selection import cross_val_predict  # For K-Fold Cross Validation
from sklearn.metrics import r2_score  # For find accuracy with R2 Score
from sklearn.metrics import mean_squared_error  # For MSE
from math import sqrt  # For squareroot operation

y_pred_DTR_test = decision_tree_reg.predict(x_test)

accuracy_DTR_test = r2_score(y_test, y_pred_DTR_test)

###########################Random Forest(case 3)###############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

%matplotlib qt

dataset = pd.read_excel('insurance.xlsx')


dataset.drop(["region"], axis=1, inplace=True)

x = dataset.drop(["charges"], axis=1)
y = dataset.charges.values  

from sklearn.preprocessing import LabelEncoder

labelencoder_sex = LabelEncoder()
x.sex = labelencoder_sex.fit_transform(x.sex)

labelencoder_smoker = LabelEncoder()
x.smoker = labelencoder_smoker.fit_transform(x.smoker)

x["bmi"] = (x - np.min(x))/(np.max(x) - np.min(x)).values

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regression model

from sklearn.model_selection import train_test_split  # Import "train_test_split" method

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


random_forest_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=13)
random_forest_reg.fit(x_train, y_train)

y_pred_RFR_test = random_forest_reg.predict(x_test)

from sklearn.model_selection import cross_val_predict  # For K-Fold Cross Validation
from sklearn.metrics import r2_score  # For find accuracy with R2 Score
from sklearn.metrics import mean_squared_error  # For MSE
from math import sqrt  # For squareroot operation

accuracy_RFR_test = r2_score(y_test, y_pred_RFR_test)