import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


# Read in data
x_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
x_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# train model
print('Training ML model...')

N_ESTIMATORS = 2
MAX_DEPTH = 2
model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)
model = model.fit(x_train, y_train)

# print accuracy
accuracy = model.score(x_test, y_test)
print(accuracy)
with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(accuracy) + "\n")

# Plot confusion matrix
disp = plot_confusion_matrix(model, x_test, y_test,cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')
plt.show()

