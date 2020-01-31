from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
import numpy as np

csv_file = 'E:\Work\Programming_Practice\Prep\Data\shoe_data.csv'
values = np.genfromtxt(csv_file, delimiter = ',', dtype ='|U')

# select model
clf = RandomForestClassifier(max_depth = 2, random_state = 0)

# data 
X = (values[1:,0:3])
#print("X =", values[1:,0:3])

# labels
Y = (values[1:,3])
#print("Y =", values[1:,3])

# train model
clf.fit(X, Y)

# make prediction
prediction = clf.predict([[180, 60, 35]])

print("prediction:", prediction)