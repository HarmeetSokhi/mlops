import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os

# load data
data = load_breast_cancer()

# preprocess data
x = pd.DataFrame(data.data, columns=data.feature_names)
print("X", x.head())

y = pd.DataFrame(data.target,)
print("Y", y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y)

# Save it
if not os.path.isdir("data"):
    os.mkdir("data")
np.savetxt("data/train_features.csv",x_train)
np.savetxt("data/test_features.csv",x_test)
np.savetxt("data/train_labels.csv",y_train)
np.savetxt("data/test_labels.csv",y_test)


