##########################Aritficial Neaural Network###########################
# =============================================================================
# importing the libraries
# =============================================================================
import pandas as pd
import numpy as np

# =============================================================================
# importing the dataset 
# =============================================================================
dataset = pd.read_excel('ChurnModelling.xlsx')
dataset1 = dataset

x = dataset1.drop(['RowNumber','CustomerId','Surname','Exited'], axis = 1)
y = dataset1.Exited

# =============================================================================
# Encoding the categorical variable #Geography, #Gender
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Gender
labelencoder_Gender = LabelEncoder()
x.Gender = labelencoder_Gender.fit_transform(x.Gender)

#Geography
labelencoder_Geography = LabelEncoder()
x.Geography = labelencoder_Geography.fit_transform(x.Geography)

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()

##############removing the dummy variable trap
x = x[:,1:]

# =============================================================================
#splitting the dataset into training set and test set 
# =============================================================================
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2 , random_state = 0)

# =============================================================================
# feature scaling
# =============================================================================

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# =============================================================================
# importing the Keras libraries
# =============================================================================
import keras
from keras.models import Sequential
from keras.layers import Dense

# =============================================================================
# initialising the ANN
# =============================================================================
classifier = Sequential()

# =============================================================================
# Adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size = 10, nb_epoch = 100)

#predicting the test set result
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

#Making the cunfusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)






























































