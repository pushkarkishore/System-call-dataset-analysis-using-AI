import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

housing = pd.read_csv("housing.csv")
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
housing.hist(bins=50, figsize=(20,15))
def split_train_test(data, test_ratio):
    shuffled_indices= np.random.permutation(len(data))
    test_set_size= int(len(data) *test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
len(test_set)

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    print(ids)
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
housing_with_id =housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"index")
train_set, test_set = train_test_split(housing,test_size=0.2,random_state=42)
housing["income_cat"]= pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
housing["income_cat"].hist()
# stratified split
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=None)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]
strat_test_set["income_cat"].value_counts()/len(strat_test_set)    
# removing attribute
for set in (strat_train_set, strat_test_set):
    set.drop("income_cat",axis=1,inplace=True)
# Copying training set
housing = strat_train_set.copy()    
# visualization of features
housing.plot(kind="scatter",x="longitude",y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", cmap= plt.get_cmap("jet"), colorbar=True)
#coorelations among features
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending= False)
scatter_matrix(housing[["median_house_value","median_income","total_rooms"]], figsize=(12,8))
#combination of attributes
housing["rooms_per_household"]= housing["total_rooms"]/housing["households"]
housing["bedrooms_per_household"]= housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_threshold"]= housing["population"]/housing["households"]
corr_matrix=housing.corr()
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
#Data cleaning (getting rid of the corresponding districts)
housing.dropna(subset = ["total_bedrooms"])
#Data cleaning (getting rid of the whole attribute)
housing.drop("total_bedrooms", axis=1)
#Data cleaning (setting values to some value)
median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)
imputer= Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X= imputer.transform(housing_num)
# Array to dataframe
housing_tr = pd.DataFrame(X,index=housing_num.index,columns = housing_num.columns)
housing_cat= housing[["ocean_proximity"]]
housing_cat.head(10)
# encoding catergorical data to numeric
label_encoder = LabelEncoder()
label_encoded =label_encoder.fit_transform(housing_cat)
label_encoded[:10]
# OneHotEncoder
cat_encoder = OneHotEncoder(dtype=np.int64)
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
# Custom Transformers
rooms_x=3
bedroms_ix=4
population_ix= 5
households_ix= 6
class CombinedAttributesAdder (BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room =add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_x] / X[:,households_ix]    
        population_per_threshold =X[:, population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedroms_ix] / X[:,rooms_x]
            return np.c_[X, rooms_per_household, population_per_threshold, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_threshold]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs= attr_adder.transform(housing.values)
num_pipeline = Pipeline([('imputer',Imputer(strategy="median")),('attribs_adder',CombinedAttributesAdder()),('std_scaler', StandardScaler()),])
housing_num_tr = num_pipeline.fit_transform(housing_num)
# Column transfer process
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([("num",num_pipeline, num_attribs), ("cat",OneHotEncoder(), cat_attribs),])
housing_prepared = full_pipeline.fit_transform(housing)
# linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("predictions:", lin_reg.predict(some_data_prepared
                                      ))
print("labels", list(some_labels))
# Error calculation in linear regression
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse= mean_squared_error(housing_labels,housing_predictions)
lim_rmse = np.sqrt(lin_mse)
# Decisiontree classifier
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# cross-validation 
scores= cross_val_score(tree_reg,housing_prepared,housing_labels,scoring ="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
tree_rmse_scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean", scores.mean())
    print("Standard deviation", scores.std())
display_scores(tree_rmse_scores)    
# cross-validation regressor
forest_reg =RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)
reg_scores= cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels,housing_predictions)
forest_rmse = np.sqrt(forest_mse)
display_scores(forest_rmse)
# Saving model and loading model
joblib.dump(forest_reg, "forest_reg.pkl")
my_model_loaded = joblib.load("forest_reg.pkl")
# Grid search
param_grid = [{'n_estimators':[3,10,30], 'max_features': [2,4,6,8]}, 
              {'bootstrap':[False], 'n_estimators':[3,10], 'max_features': [2,3,4]},]
forest_reg = RandomForestRegressor()
grid_search =GridSearchCV(forest_reg, param_grid,cv =5, scoring='neg_mean_squared_error',return_train_score= True)
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)
# Analyzing the best models and their errors
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs =["rooms_per_household","population_per_threshold","bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances,attributes), reverse=True)
# Evaluating your system on the test set
final_model = grid_search.best_estimator_
X_test= strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
    
    