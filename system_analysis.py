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
api_calls = pd.read_csv('C:\\Users\\pshkr\\Downloads\\call_2\\mal-api-2019 (3)\\data_nn.csv')
train_set, test_set = train_test_split(api_calls, test_size=0.2, random_state=42)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(api_calls,api_calls["Types"]):
    strat_train_set = api_calls.loc[train_index]
    strat_test_set = api_calls.loc[test_index]
housing = strat_train_set.copy()    
housing = strat_train_set.drop("Types", axis=1) 
avast = housing_prepared_nn.values.reshape((5685, 109, 109))# neural networks
housing_labels = strat_train_set["Types"].copy()
X_test = strat_test_set.drop("Types", axis=1)
housing_prepared_test_reg = num_pipeline.fit_transform(X_test)# neural networks
housing_prepared_test_reg = pd.DataFrame(housing_prepared_test_reg)# neural networks
housing_prepared_test = num_pipelines.fit_transform(X_test)# neural networks
housing_prepared_test = pd.DataFrame(housing_prepared_test)# neural networks
avast_test = housing_prepared_test.values.reshape((1422, 109, 109))# neural networks
y_test = strat_test_set["Types"].copy()
# sequential classification API neural network
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[109,109]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(9, activation="softmax"))
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
history = model.fit(avast, housing_labels, epochs=100,
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
# Glorot and He initialization
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")
he_avg_int = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
                                          distribution='uniform')
keras.layers.Dense(10, activation="relu", kernel_initializer=he_avg_int)
# Leaky Relu
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"])
history = model.fit(avast,housing_labels, epochs=100,
                    validation_split=0.1)
model.evaluate(avast_test,y_test)
# PRELU
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"])
history = model.fit(avast,housing_labels, epochs=200,
                    validation_split=0.1,callbacks=[tensorboard_cb])
model.evaluate(avast_test,y_test)
#SRELU
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[109, 109]))
model.add(keras.layers.Dense(300, activation="selu",
                             kernel_initializer="lecun_normal"))
for layer in range(99):
    model.add(keras.layers.Dense(100, activation="selu",
                                 kernel_initializer="lecun_normal"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"])
history = model.fit(avast, housing_labels, epochs=100,validation_split=0.1)
model.evaluate(avast_test,y_test)
# Batch normalization
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()
bn1 = model.layers[1]
[(var.name, var.trainable) for var in bn1.variables]
model.layers[1].updates
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(100, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"])
history = model.fit(avast, housing_labels, epochs=100,validation_split=0.1)
model.evaluate(avast_test,y_test)     
# REUSING PRETRAINED LAYERS
def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2 # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32) # 
    classification task: is it a shirt (class 6)?
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))
(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(avast, housing_labels)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(avast_test, y_test)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(),
                metrics=["accuracy"])
history = model.fit(X_train_A, y_train_A, epochs=100,
                    validation_split=0.1)
model.save("my_model_A.h5")
model.evaluate(X_test_A,y_test_A)         
model_B = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(1, activation="sigmoid")
])
model_B.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.SGD(),
                metrics=["accuracy"])
history = model_B.fit(X_train_B, y_train_B, epochs=100,validation_split=0.1)
model_B.evaluate(X_test_B,y_test_B)                      
model_A = keras.models.load_model("my_model_A.h5")    
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])         
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(),
                     metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=100,
                           validation_split=0.1)
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True
model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3),
                     metrics=["accuracy"])    
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=100,
                           validation_split=0.1)
model_B.evaluate(X_test_B, y_test_B)
model_B_on_A.evaluate(X_test_B, y_test_B)
# momentum optimization
optimizer= keras.optimizers.SGD(lr=0.001, momentum=0.9)
#Nesterov accelerated Gradient
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
#Adagrad
optimizer = keras.optimizers.Adagrad(lr=0.001)
#RMSProp
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
# Adam optimization
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
#Adamax
optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)
#nadam optimization
optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
# POWER SCHEDULING
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(avast,housing_labels, epochs=100,
                    validation_split=0.1)
model.evaluate(avast_test,y_test)
# Exponential Scheduling
def exponential_decay_fn(epoch):
    return 0.01 * 0.1**(epoch / 20)
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn
exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"])
n_epochs=100
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(avast, housing_labels, epochs=n_epochs,
                   validation_split=0.1,
                    callbacks=[lr_scheduler])
model.evaluate(avast_test,y_test)
# Piecewise constant scheduling
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
def piecewise_constant(boundaries, values):
    boundaries = np.array([0] + boundaries)
    values = np.array(values)
    def piecewise_constant_fn(epoch):
        return values[np.argmax(boundaries > epoch) - 1]
    return piecewise_constant_fn

piecewise_constant_fn = piecewise_constant([5, 15], [0.01, 0.005, 0.001])   
lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"]) 
n_epochs=100    
history = model.fit(avast, housing_labels, epochs=n_epochs,
                   validation_split=0.1,
                    callbacks=[lr_scheduler])
model.evaluate(avast_test,y_test)
# performance scheduling
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])
optimizer = keras.optimizers.SGD(lr=0.02, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
n_epochs = 100
history = model.fit(avast, housing_labels, epochs=n_epochs,
                   validation_split=0.1,
                    callbacks=[lr_scheduler])
model.evaluate(avast_test,y_test)
# tf.keras scheduler
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])
s = 20 * len(avast) // 32 # number of steps in 20 epochs (batch size = 32)
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
n_epochs = 100
history = model.fit(avast, housing_labels, epochs=n_epochs,
                   validation_split=0.1
                    )
model.evaluate(avast_test,y_test)
# 1-cycle scheduling
K = keras.backend
class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
    init_weights = model.get_weights()
    iterations = len(X) // batch_size * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
]) 
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"]) 
batch_size = 128
rates, losses = find_learning_rate(model, avast, housing_labels, epochs=100, batch_size=batch_size)
plot_lr_vs_loss(rates, losses)  
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)
n_epochs = 100
onecycle = OneCycleScheduler(len(avast) // batch_size * n_epochs, max_rate=0.05)
history = model.fit(avast, housing_labels, epochs=n_epochs, batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[onecycle])        
model.evaluate(avast_test,y_test)

# l1 and l2 regularization
layer = keras.layers.Dense(100, activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))
from functools import partial
RegularizedDense = partial(keras.layers.Dense,
                           activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(), metrics=["accuracy"])
n_epochs = 100
history = model.fit(avast,housing_labels, epochs=n_epochs,
                   validation_split=0.1)
model.evaluate(avast_test,y_test)
# Dropout
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(), metrics=["accuracy"])
n_epochs = 100
history = model.fit(avast, housing_labels, epochs=n_epochs,
                    validation_split=0.1)
model.evaluate(avast_test,y_test)
# alpha dropout
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
n_epochs = 100
history = model.fit(avast, housing_labels, epochs=n_epochs,
                    validation_split=0.1)
y_pred=model.evaluate(avast_test,y_test)
# mc dropout
y_probas = np.stack([model(avast, training=True)
                     for sample in range(100)])
y_proba = y_probas.mean(axis=0)
y_std = y_probas.std(axis=0)
np.round(model.predict(avast_test[:1]), 2)
np.round(y_probas[:, :1], 2)
np.round(y_proba[:1], 2)
model.evaluate(avast_test,y_test)
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
mc_model = keras.models.Sequential([
    MCAlphaDropout(layer.rate) if isinstance(layer, keras.layers.AlphaDropout) else layer
    for layer in model.layers
])    
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
mc_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
mc_model.set_weights(model.get_weights())
history = mc_model.fit(avast, housing_labels, epochs=100,
                    validation_split=0.1,callbacks=[tensorboard_cb])
# max norm
layer = keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal",
                           kernel_constraint=keras.constraints.max_norm(1.))
MaxNormDense = partial(keras.layers.Dense,
                       activation="selu", kernel_initializer="lecun_normal",
                       kernel_constraint=keras.constraints.max_norm(1.))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    MaxNormDense(300),
    MaxNormDense(100),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
n_epochs = 100
history = model.fit(avast, housing_labels, epochs=n_epochs,
                     validation_split=0.1,callbacks=[tensorboard_cb])
# Tensors and Operations
t= tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
tf.constant(42) # scalar
t.shape
t.dtype
# tensor indexing
t[:, 1:]
t[..., 1, tf.newaxis]
# tensor operations
t + 10
tf.square(t)
t @ tf.transpose(t)
# Using keras.backend
from tensorflow import keras
K = keras.backend
K.square(K.transpose(t)) + 10
# similarity with numpy
a = np.array([2., 4., 5.])
tf.constant(a)
t.numpy()
np.array(t)
tf.square(a)
np.square(t)
# conflicting types
t2 = tf.constant(40., dtype=tf.float64)
tf.constant(2.0) + tf.cast(t2, tf.float32)
# variables
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
v.assign(2 * v)
v[0, 1].assign(42)
v[:, 2].assign([0., 1.])
v.scatter_nd_update(indices=[[0, 0], [1, 2]],
                    updates=[100., 200.])
# sparse tensors
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])
print(s)
tf.sparse.to_dense(s)
#Tensor Arrays
array = tf.TensorArray(dtype=tf.float32, size=3)
array = array.write(0, tf.constant([1., 2.]))
array = array.write(1, tf.constant([3., 10.]))
array = array.write(2, tf.constant([5., 7.]))
array.read(1)
array.stack()
#Ragged Tensors
p = tf.constant(["Café", "Coffee", "caffè", "break"])
tf.strings.length(p, unit="UTF8_CHAR")
r = tf.strings.unicode_decode(p, "UTF8")
print(r)
r2 = tf.ragged.constant([[65, 66], [], [67]])
print(tf.concat([r, r2], axis=0))
# Custom Loss Functions
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss=huber_fn, optimizer=optimizer,metrics=["accuracy"])
history = model.fit(avast, housing_labels, epochs=10,
                     validation_split=0.1,callbacks=[tensorboard_cb])
model.evaluate(avast_test,y_test)
# saving and loading custom component model
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])    
model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["accuracy"])
history = model.fit(avast, housing_labels, epochs=10,
                     validation_split=0.1,callbacks=[tensorboard_cb])
model.save("my_model_with_a_custom_loss_class.h5")
model = keras.models.load_model("my_model_with_a_custom_loss_class.h5",custom_objects={"HuberLoss": HuberLoss})
history = model.fit(avast, housing_labels, epochs=10,
                     validation_split=0.1,callbacks=[tensorboard_cb])   
# other custom functions
def my_softplus(z): # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)
def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)
def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))
def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights) 
layer = keras.layers.Dense(1, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)
keras.backend.clear_session()
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation=my_softplus,
                       kernel_regularizer=MyL1Regularizer(0.01),
                       kernel_constraint=my_positive_weights,
                       kernel_initializer=my_glorot_initializer),
])    
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"])
history = model.fit(avast, housing_labels, epochs=100,
                     validation_split=0.1,callbacks=[tensorboard_cb])
model.save("my_model_with_many_custom_parts.h5")
model = keras.models.load_model(
    "my_model_with_many_custom_parts.h5",
    custom_objects={
       "MyL1Regularizer": MyL1Regularizer,
       "my_positive_weights": my_positive_weights,
       "my_glorot_initializer": my_glorot_initializer,
       "my_softplus": my_softplus,
    })
# Custom metrics
keras.backend.clear_session()
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])  
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(), metrics=[create_huber(2.0)])
history = model.fit(avast, housing_labels, epochs=100,
                     validation_split=0.1,callbacks=[tensorboard_cb])
model.evaluate(avast_test,y_test)
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.threshold = threshold
        #self.huber_fn = create_huber(threshold) # TODO: investigate why this fails
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    def huber_fn(self, y_true, y_pred): # workaround
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    def result(self):
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
model.compile(loss=create_huber(2.0), optimizer=keras.optimizers.SGD(), metrics=[HuberMetric(2.0)])
history = model.fit(avast, housing_labels, epochs=100,
                     validation_split=0.1,callbacks=[tensorboard_cb])
model.save("my_model_with_a_custom_metric.h5")
model = keras.models.load_model("my_model_with_a_custom_metric.h5",         
                         custom_objects={"huber_fn": create_huber(2.0),"HuberMetric": HuberMetric})
#Custom Layers
keras.backend.clear_session()
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape) # must be at the end

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109, 109]),
    MyDense(30, activation="relu"),
    MyDense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(), metrics=["accuracy"])
history = model.fit(avast, housing_labels, epochs=100,
                     validation_split=0.1)
model.evaluate(avast_test,y_test)
# Custom Models
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                          kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z
class ResidualClassifier(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.flatten1 = keras.layers.Flatten(input_shape=[109,109])
        self.hidden1 = keras.layers.Dense(300, activation="elu",
                                          kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 300)
        self.block2 = ResidualBlock(2, 300)
        self.out = keras.layers.Dense(output_dim,activation="softmax")

    def call(self, inputs):
        Z=self.flatten1(inputs)
        Z = self.hidden1(Z)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        
        return self.out(Z)    
model = ResidualClassifier(10)    
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(), metrics=["accuracy"])
history = model.fit(avast, housing_labels, epochs=100,
                     validation_split=0.1)
model.evaluate(avast_test,y_test)
model.summary()
# losses and metrics based on model internals [need refinement for classification]
class ReconstructingRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.flatten1 = keras.layers.Flatten(input_shape=[109,109])
        self.hidden = [keras.layers.Dense(300, activation="relu",
                                          kernel_initializer="lecun_normal")
                       for _ in range(5)]
        self.out = keras.layers.Dense(output_dim,activation="softmax")

    def build(self, batch_input_shape):
       
        n_inputs = batch_input_shape[-1]
        print(n_inputs)
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)

    def call(self, inputs, training=None):
        Z=self.flatten1(inputs)
        
        for layer in self.hidden:
            Z = layer(Z)
         
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        #if training:
        #    result = self.reconstruction_mean(recon_loss)
        #    self.add_metric(result)
        return self.out(Z)
model = ReconstructingRegressor(10)
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(), metrics=["accuracy"])
history = model.fit(avast, housing_labels, epochs=100,validation_split=0.1)  
model.summary()
model.layers
# Computing Gradients Using Autodiff
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)
dz_dw1 = tape.gradient(z, w1)
dz_dw2 = tape.gradient(z, w2) 
del tape
c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)
    z= f(c1,c2)
gradients = tape.gradient(z, [c1, c2])
def f(w1, w2):
    return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)
with tf.GradientTape() as tape:
    z = f(w1, w2)
tape.gradient(z, [w1, w2])
@tf.custom_gradient
def my_better_softplus(z):
    exp = tf.exp(z)
    def my_softplus_gradients(grad):
        return grad / (1 + 1 / exp)
    return tf.math.log(exp + 1), my_softplus_gradients
def my_better_softplus(z):
    return tf.where(z > 30., z, tf.math.log(tf.exp(z) + 1.))
x = tf.Variable([15.])
with tf.GradientTape() as tape:
    z = my_better_softplus(x)
z, tape.gradient(z, [x])
# Custom Training Loops [shape issue between y_pred and y_batch]
l2_reg = keras.regularizers.l2(0.05)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[109,109]),
    keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                       kernel_regularizer=l2_reg),
    keras.layers.Dense(10,activation="softmax")  
])
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size) 
    return X[idx], y[idx]
def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                         for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics,
          end=end)

n_epochs=100
batch_size = 32
n_steps = len(avast) // batch_size
optimizer = keras.optimizers.Nadam(lr=0.01)
housing_labels = housing_labels.to_numpy()
metricsAcc = tf.keras.metrics.Accuracy()

for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(avast, housing_labels)
        
        with tf.GradientTape() as tape:
            y_pred= model(X_batch,training=True)
            loss_values=tf.keras.losses.sparse_categorical_crossentropy(y_batch, y_pred)
            
        gradients = tape.gradient(loss_values, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        metricLoss=tf.keras.metrics.sparse_categorical_crossentropy(y_batch, y_pred)
        metricsAcc.update_state(y_batch,y_pred)
        readout = 'Epoch {}, Training loss: {}, Training accuracy: {}'
        print(readout.format(epoch + 1, loss_values,
                              metricsAcc.result() * 100))
        metricsAcc.reset_states
#  Tensorflow functions and graphs
def cube(x):
    return x ** 3   
tf.cube=tf.function(cube)   
tf.cube(2)  



    

 
