import sklearn
import pandas as pd
import sklearn.neighbors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import os
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
# To plot pretty figures directly within spyder
%matplotlib inline
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
# Load the data
oecd_bli = pd.read_csv('training_newupdated.csv')
oecd_bli.head()
oecd_bli.info()
oecd_bli["666 170 148"].value_counts()
oecd_bli.describe()
# to make this notebook's output identical at every run
np.random.seed(42)
# For illustration only. Sklearn has train_test_split()
train_set, test_set = train_test_split(oecd_bli, test_size=0.2, random_state=42)
test_set.head()
# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(oecd_bli,oecd_bli["Types"]):
    strat_train_set = oecd_bli.loc[train_index]
    strat_test_set = oecd_bli.loc[test_index]
# Looking for correltaions
housing = strat_train_set.copy()
corr_matrix = housing.corr()
corr_data=corr_matrix["Types"].sort_values(ascending=False)
attributes = ["259 304 259", "Types"]
scatter_matrix(oecd_bli[attributes], figsize=(12, 8))
# drop labels for training data
housing = strat_train_set.drop("Types", axis=1) 
housing_labels = strat_train_set["Types"].copy()
# transformation pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        
        ('std_scaler', StandardScaler())
    ])
housing.drop(housing.columns[[0,1]], axis=1, inplace=True)
housing_prepared = num_pipeline.fit_transform(housing)
# Visualize the data
oecd_bli.plot(kind='scatter', x="666 170 148", y="Types")
# Select a linear regression model
model = sklearn.linear_model.LinearRegression()
# Train the model
model.fit(housing, housing_labels)
# Make a prediction
predictions= model.predict(housing)  
# Error calculation
lin_scores = cross_val_score(model, housing, housing_labels,
                             scoring="neg_mean_squared_error", cv=2)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
# Select a decision tree regressor model
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing,housing_labels)
predictions = tree_reg.predict(housing)
# Error calculation
scores = cross_val_score(tree_reg, housing, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
# select a randomforestregressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing, housing_labels)
predictions = forest_reg.predict(housing)
forest_scores = cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
# Select a Kneighbours regression model
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
model.fit(X,y)  
predictory = model.predict(X)  
# Grid search
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
grid_search = GridSearchCV(forest_reg, param_grid, cv=2,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing, housing_labels)
grid_search.best_params_
grid_search.best_estimator_
# Feature importance
feature_importances = grid_search.best_estimator_.feature_importances_
# Test on the test set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("Types", axis=1)
y_test = strat_test_set["Types"].copy()
X_test.drop(X_test.columns[[0,1]], axis=1, inplace=True)
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
