# importing data handling libraries and ml libraries
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# importing the clean data to python
file = r'C:\Users\Todimu-PC\Downloads\soilData.xlsx'
data = pd.read_excel(file)

# removing outliers from the dataset
z_scores = zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data_new = data[filtered_entries]

# splitting the data into target and predictors
X = data_new.drop('compression index (C0)', axis=1)
y = data_new['compression index (C0)']

# scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.3, random_state=42)

# building the machine learning model and scaling the data.
forest = RandomForestRegressor()
pipeReg = Pipeline([('Scaler', StandardScaler()), ('regressor', RandomForestRegressor())])
pipeReg.fit(X_train, y_train)

# making predictions
yPredF = pipeReg.predict(X_test)
mean_squared_error(y_test, yPredF)

print(pipeReg.score(X_test, y_test))

# saving the model as a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeReg, f)
