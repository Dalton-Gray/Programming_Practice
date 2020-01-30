from sklearn import tree
import numpy as np

# import data 
csv_file = 'E:\Work\Programming_Practice\Prep\Data\Melbourne_housing_FULL.csv'
values = np.genfromtxt(csv_file, delimiter = ',', dtype ='|U')