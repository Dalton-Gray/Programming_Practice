from sklearn import tree 
import numpy as np

# import data
csv_file = 'E:\Work\Programming_Practice\Prep\Data\shoe_data.csv'
values = np.genfromtxt(csv_file, delimiter = ',', dtype ='|U')

# select model
clf = tree.DecisionTreeClassifier()

# data 
X = (values[1:,3])

# labels
Y = (values[0])

# train model
clf = clf.fit(X, Y)

# make prediction
prediction = clf.predict([[190, 70, 43]])

print(prediction)
