import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors
# Load the data
oecd_bli= pd.read_csv("oecd_bli_2015.csv", thousands =',')
gdp_per_capita= pd.read_csv("gdp_per_capita.csv", thousands=',',delimiter='\t',encoding='latin1', na_values="n/a")
# Redefining dataset features
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
# Prepare the data
country_stats= prepare_country_stats(oecd_bli, gdp_per_capita)
# Visualize the data
country_stats.plot(kind ='scatter',x='GDP per capita', y='Life satisfaction')
# selecting a model for linear regression and KNeighboursRegressor
model = sklearn.linear_model.LinearRegression()
model2= sklearn.neighbors.KNeighborsRegressor(n_neighbors = 3)
#Train a model
X= np.c_[country_stats["GDP per capita"]]
y= np.c_[country_stats["Life satisfaction"]]
model.fit(X, y)
model2.fit(X,y)
# make prediction
model2.score(X,y)
model.predict([[22587]])
model2.predict([[22587]])
