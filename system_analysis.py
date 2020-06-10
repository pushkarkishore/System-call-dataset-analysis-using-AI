import sklearn
import tensorflow as tf
from tensorflow import keras
tf.__version__
keras.__version__
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
from sklearn.preprocessing import MinMaxScaler
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
from sklearn.linear_model import Perceptron
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import os
root_logdir = os.path.join(os.curdir, "my_logs")
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
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
housing_labels_0 = (housing_labels==0) # if classification is required
# transformation pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        
        ('std_scaler', StandardScaler())
    ])
num_pipelines = Pipeline([
        
        ('min_scaler', MinMaxScaler())
    ]) # neural networks dataset
housing.drop(housing.columns[[0,1]], axis=1, inplace=True)
housing_prepared = num_pipeline.fit_transform(housing)
housing_prepared = pd.DataFrame(housing_prepared)
housing_prepared_nn = num_pipelines.fit_transform(housing)# neural networks
housing_prepared_nn = pd.DataFrame(housing_prepared_nn)# neural networks
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
# select a perceptron
per_clf = Perceptron()
per_clf.fit(housing,housing_labels_0)
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
y_pred = per_clf.predict(X_test) # prediction for perceptron
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
confusion_matrix(y_pred,y_test_5)# for perceptron
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
avast = housing_prepared_nn.values.reshape((6476, 109, 109))# neural networks
housing_labels = strat_train_set["Types"].copy()
X_test = strat_test_set.drop("Types", axis=1)
housing_prepared_test_reg = num_pipeline.fit_transform(X_test)# neural networks
housing_prepared_test_reg = pd.DataFrame(housing_prepared_test_reg)# neural networks
housing_prepared_test = num_pipelines.fit_transform(X_test)# neural networks
housing_prepared_test = pd.DataFrame(housing_prepared_test)# neural networks
avast_test = housing_prepared_test.values.reshape((1620, 109, 109))# neural networks
y_test = strat_test_set["Types"].copy()
# sequential classification API neural network
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[109,109]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(16, activation="softmax"))
# sequential regression API neural network
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=housing_prepared_nn.shape[1:]),
    keras.layers.Dense(1)
]) 
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(housing_prepared_nn, housing_labels, epochs=20, validation_split=0.1 )
model.evaluate(housing_prepared_test, y_test)
y_pred = model.predict(housing_prepared_test)
# building complex models using function API
input_ = keras.layers.Input(shape=housing_prepared_nn.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_], outputs=[output])
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(housing_prepared_nn, housing_labels, epochs=20, validation_split=0.1)
model.evaluate(housing_prepared_test, y_test)
y_pred = model.predict(housing_prepared_test)
# building complex models using function API with 2 inputs
input_A = keras.layers.Input(shape=[1800], name="wide_input")
input_B = keras.layers.Input(shape=[2126], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
model.compile(loss="mse", optimizer=keras.optimizers.SGD())
X_train_A, X_train_B = housing_prepared_nn.iloc[:, :1800], housing_prepared_nn.iloc[:, 1801:]
X_test_A, X_test_B = housing_prepared_test.iloc[:, :1800], housing_prepared_test.iloc[:, 1801:]
history = model.fit((X_train_A, X_train_B), housing_labels, epochs=20, validation_split=0.1)
model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_test_A,X_test_B))
# using subclassing api to build dynamic models
class WideAndDeepModel(keras.models.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
        
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output
model = WideAndDeepModel(30, activation="relu")
# model summary for neural network 
model.summary()
model.layers
hidden1 = model.layers[1]
hidden1.name
model.get_layer(hidden1.name) is hidden1
weights, biases = hidden1.get_weights()
weights
weights.shape
biases
biases.shape
# compile model for sequential classification neural network
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
# train and evaluate the sequential classification neural network model
history = model.fit(avast, housing_labels, epochs=80,
                    validation_split=0.1)
model.evaluate(avast_test,y_test)
y_pred = model.predict_classes(avast_test)
# SVM
svm_clf= SVC()
svm_clf.fit(housing,housing_labels)
svm_clf.predict(X_test)
# One vs Rest classifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(housing,housing_labels)
predictions= ovr_clf.predict(X_test)
y_train_pred= cross_val_predict(sgd_clf, housing, housing_labels, cv=3)
conf_mx = confusion_matrix(housing_labels, y_train_pred)
# plotting results
plt.matshow(conf_mx, cmap=plt.cm.gray)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
# using callbacks to save long training model
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
history = model.fit(housing_prepared_nn, housing_labels, epochs=100, validation_split=0.1, callbacks=[checkpoint_cb, early_stopping_cb])
# Using tensorboard for visualization
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(housing_prepared_nn, housing_labels, epochs=30, validation_split=0.1, callbacks=[tensorboard_cb])
# fine tuning neural netowrk hyperparameter
def build_model(n_neurons_1=300, n_neurons_2=100, learning_rate=3e-3, input_shape=[109,109]):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(n_neurons_1, activation="relu"))
    model.add(keras.layers.Dense(n_neurons_2, activation="relu"))
    model.add(keras.layers.Dense(16, activation="softmax"))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,  metrics = ["accuracy"])
   
    return model
keras_claf = keras.wrappers.scikit_learn.KerasClassifier(build_model)
keras_claf.fit(avast,housing_labels,epochs=100,validation_split=0.1,callbacks=[keras.callbacks.EarlyStopping(patience=10)] )
mse_test = keras_claf.score(avast_test,y_test)
y_pred = keras_claf.predict(avast_test)
param_distribs = {
    "n_neurons_1": np.arange(200, 400),
    "n_neurons_2": np.arange(50, 200),
    "learning_rate": reciprocal(3e-4, 3e-2),
}
rnd_search_cv = RandomizedSearchCV(keras_claf, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(avast, housing_labels, epochs=100,
                  validation_split=0.1,
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
rnd_search_cv.best_params_
rnd_search_cv.best_score_
