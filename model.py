import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import datasets
data = datasets.load_diabetes()

df = pd.DataFrame(data.data)
df.columns = data.feature_names
df['target'] = data.target

x = df[["age","sex","bmi","bp"]]
y = df[["target"]]

regressor = LinearRegression()
regressor.fit(x, y)

import pickle
pickle.dump(regressor, open('model.pkl','wb'))