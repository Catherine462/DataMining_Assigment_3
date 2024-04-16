import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# 11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

# Reading the training data
df_training = pd.read_csv('weather_training.csv')
X_train = df_training.iloc[:, 1:].values  
y_train = df_training.iloc[:, -1].values

# Update the training class values according to the discretization (11 values only)
y_train_discretized = np.digitize(y_train, classes) - 1

# Reading the test data
df_test = pd.read_csv('weather_test.csv')
X_test = df_test.iloc[:, 1:].values  
y_test = df_test.iloc[:, -1].values

# Update the test class values according to the discretization (11 values only)
y_test_discretized = np.digitize(y_test, classes) - 1

# Fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_train, y_train_discretized)

# Make the naive_bayes prediction for each test sample and start computing its accuracy
predictions = clf.predict(X_test)
accurate_count = 0
total_count = len(y_test_discretized)
for pred, actual in zip(predictions, y_test_discretized):
    if abs(pred - actual) <= 0.15 * abs(actual):  
        accurate_count += 1

# Calculate accuracy
accuracy = accurate_count / total_count

# Print the naive_bayes accuracy
print("naive_bayes accuracy: " + str(accuracy))
