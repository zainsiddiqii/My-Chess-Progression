# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# set test size and random state (for reproducilibity)
test_size = 0.2
random_state = 82 

# load dataset
dataset = pd.read_csv('data/ML_processed.csv')

# split dataset into features and target
X = dataset.drop(columns = ['win'])
y = dataset.loc[:, ['win']]

# split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

# flatten target arrays to feed into models
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

print('Dataset size:', dataset.shape[0])
print('Train set size:', X_train.shape[0])
print('Test set size:', X_test.shape[0])

columns = X.columns.to_list()
dict_format = {column: 'value' for column in columns}

print('To make predictions, please provide a dictionary of values in the following format:\n', dict_format)