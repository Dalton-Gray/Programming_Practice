# import libraties 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# import data 
data_frame = pd.read_csv('E:\Work\Programming_Practice\Prep\Data\Melbourne_housing_FULL.csv')

# The misspellings of "longitude" and "latitude" are preserved, as the two misspellings were not corrected in the source file.
# delete unneeded columns 
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

# convert non numeric values to numeric using one-hot encoding
features_data_frame = pd.get_dummies(data_frame, columns =['Suburb', 'CouncilArea', 'Type'])

# removes dependent variable
del features_data_frame['Price']

# create X and Y arrays from the dataset
X = features_data_frame.values

Y = data_frame['Price'].values

# split data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, shuffle = True)

# select algorithm
model = ensemble.GradientBoostingRegressor()

# choose configurations to test
param_grid = {
	'n_estimators' : [300, 600],
	'max_depth': [7, 9],
	'min_samples_split': [3, 4],
	'min_samples_leaf': [4, 6],
	'learning_rate': [0.01, 0.02],
	'max_features': [0.8, 0.9],
	'loss': ['ls', 'lad', 'huber']
}

# define grid sreach. n_jobs = number of cores, -1 = all cores
gs_cv = GridSearchCV(model, param_grid, n_jobs = -1)


# run grid search on training data
gs_cv.fit(X_train, Y_train)

# print optimal hyperparameters 
print(gs_cv.best_params_)

# check model accuracy (up to 2 decimal places)
mse = mean_absolute_error(Y_train, gs_cv.predict(X_train))
print("Training data mean absolute error: %.2f" % mse)

# test model
mse = mean_absolute_error(Y_test, gs_cv.predict(X_test))
print("Test data mean absolute error: %.2f" % mse)
