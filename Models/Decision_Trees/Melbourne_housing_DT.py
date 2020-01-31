from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

# import data 
data_frame = pd.read_csv('E:\Work\Programming_Practice\Prep\Data\Melbourne_housing_FULL.csv')

# The misspellings of "longitude" and "latitude" are preserved, as the two misspellings were not corrected in the source file.
del data_frame['Address']
del data_frame['Method']
del data_frame['SellerG']
del data_frame['Date']
del data_frame['Postcode']
del data_frame['Lattitude']
del data_frame['Longtitude']
del data_frame['Regionname']
del data_frame['Propertycount']

# remove columns with empty values
data_frame.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)

# Convert non numeric values to numeric using one-hot encoding
features_data_frame = pd.get_dummies(data_frame, columns =['Suburb', 'CouncilArea', 'Type'])

# Removes dependent variable
del features_data_frame['Price']

# Independant variables
X = features_data_frame.values

# Dependent variables
Y = data_frame['Price'].values

# Split data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, shuffle = True)

# Select model
clf = tree.DecisionTreeClassifier()

# Train model
clf.fit(X_train, Y_train)

# Evaluate results (training)
mse = mean_absolute_error(Y_train, clf.predict(X_train))

print("Training set mean absolute error : %.2f" % mse)

# Evaluate results (test)
mse = mean_absolute_error(Y_test, clf.predict(X_test))

print("Test set mean absolute error: %.2f" % mse)