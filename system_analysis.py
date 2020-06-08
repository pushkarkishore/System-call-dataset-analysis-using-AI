import sklearn
import pandas as pd
import sklearn.neighbors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import os
import hashlib
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
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
some_digit= housing.iloc[0]
some_digit_image = some_digit.reshape(62,62)
housing_labels = strat_train_set["Types"].copy()
housing_labels_0 = (housing_labels==0)
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
                             scoring="neg_mean_squared_error", cv=3)
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
# Select a SGDClassifier model
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(housing,housing_labels_0)
sgd_predictions= sgd_clf.predict(X_test)
# select a randomforestclassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, housing, housing_labels_0, cv=3,
                                    method="predict_proba")
y_scores_forest= y_probas_forest[:, 1]
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
y_test_5= (y_test==0)
X_test.drop(X_test.columns[[0,1]], axis=1, inplace=True)
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# performance evaluation using confusion matrix
y_scores= cross_val_predict(sgd_clf, housing,housing_labels_0, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(housing_labels_0, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
threshold_90_precision= thresholds[np.argmax(precisions >= 0.95)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
confusion_matrix(housing_labels_0,y_train_pred_90)
precision_score(housing_labels_0,y_train_pred_90)
recall_score(housing_labels_0,y_train_pred)
f1_score(housing_labels_0,y_train_pred)
# performance evaluation using the ROC Curve
fpr, tpr, thresholds = roc_curve(housing_labels_0, y_scores)
fpr_forest, tpr_forest, thresholds_forest = roc_curve(housing_labels_0,y_scores_forest) # for random forest
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
plot_roc_curve(fpr, tpr)   
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest") # for random forest
roc_auc_score(housing_labels_0, y_scores)
roc_auc_score(housing_labels_0, y_scores) # for random forest
#  muticlass classification
api_calls = pd.read_csv('C:\\Users\\pshkr\\Downloads\\call_2\\mal-api-2019 (3)\\data_2_api.csv')
train_set, test_set = train_test_split(api_calls, test_size=0.2, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(api_calls,api_calls["Types"]):
    strat_train_set = api_calls.loc[train_index]
    strat_test_set = api_calls.loc[test_index]
housing = strat_train_set.copy()    
housing = strat_train_set.drop("Types", axis=1) 
housing_labels = strat_train_set["Types"].copy()
X_test = strat_test_set.drop("Types", axis=1)
y_test = strat_test_set["Types"].copy()
svm_clf= SVC()
svm_clf.fit(housing,housing_labels)
svm_clf.predict(X_test)
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(housing,housing_labels)
predictions= ovr_clf.predict(X_test)
y_train_pred= cross_val_predict(sgd_clf, housing, housing_labels, cv=3)
conf_mx = confusion_matrix(housing_labels, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
