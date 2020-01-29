####################################################################################################################

#------------------------------Logistic Regression(case 1)---------------------
# =============================================================================
# Importing the libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

# =============================================================================
# Importing the dataset
# =============================================================================
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# =============================================================================
# Splitting the dataset into training set and testset
# =============================================================================
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# =============================================================================
# Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# =============================================================================
# fitting Logistic regression to training set
# =============================================================================
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

# =============================================================================
# prediction the test set result
# =============================================================================
y_pred = classifier.predict(x_test)

# =============================================================================
# making the cunfusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
# =============================================================================
# Visualising the training set result
# =============================================================================
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# =============================================================================
# Visualising the Test set results
# =============================================================================
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# =============================================================================
# -----------------------Logistic regression (case 2)--------------------------
# =============================================================================
# =============================================================================
# importing the libraries
# =============================================================================
import pandas as pd
import numpy as np
import math   #to calculate basic mathematical function
import matplotlib.pyplot as plt
import seaborn as sns    #for statistical plotting 
%matplotlib qt

dataset = pd.read_csv('titanic.csv')

# =============================================================================
# analysing data
# =============================================================================

sns.countplot(x = "Survived", data = dataset) #to display who survived or who not survived

sns.countplot(x = "Survived", hue = "Sex", data = dataset) #to display how many m/f survived

sns.countplot(x = "Survived", hue = "Pclass", data = dataset) #to display the record of how many passenger travelling in which class

dataset["Age"].plot.hist() #to display the ages of the pessengr working on the titanic

dataset["Fare"].plot.hist(bins = 20, figsize = (10,5)) #to display the fare of the passenger

sns.countplot(x = "SibSp", data = dataset)  #to display the record of the sibbling of the passenger

# =============================================================================
# Data wrangling
# =============================================================================

dataset.isnull()   #to display the null values (true = null, false = not null)

dataset.isnull().sum()   #to display the aggregate clomns which are having the null values

sns.heatmap(dataset.isnull(), yticklabels = False, cmap = "viridis")   #heatmap to display the null colomns values

dataset.drop("Cabin", axis = 1, inplace = True) #to drop the "cabin" colomns because no use of it

dataset.dropna(inplace = True)   #to drop the na values

sex = pd.get_dummies(dataset["Sex"], drop_first=True)  #to convert the string variable(sex) to the categorical variable and remove the female because 1 = male, 0 = female

embark = pd.get_dummies(dataset["Embarked"],drop_first = True) #same which done with the sex

pcl = pd.get_dummies(dataset["Pclass"],drop_first = True) #same which done with the sex

dataset = pd.concat([dataset,sex,embark,pcl],axis = 1)  #to concat all the altered variable to the origina;l

dataset.drop(["Sex","Embarked","PassengerId","Name","Ticket", "Pclass"], axis = 1, inplace = True)  #dropping the useless colomns

# =============================================================================
# train data
# =============================================================================
x = dataset.drop("Survived", axis = 1)
y = dataset["Survived"]

# =============================================================================
# splitting the dataset into training set and test set
# =============================================================================
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)

prediction = logmodel.predict(x_test)

from sklearn.metrics import classification_report

classification_report(y_test,prediction) #to calculate the accuracy and the peformance

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,prediction) # to display the accuracy

    from sklearn.metrics import accuracy_score

accuracy_score(y_test,prediction) #to display the accuracy in percentage
