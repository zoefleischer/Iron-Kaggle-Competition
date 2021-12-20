# ---------------- IMPORTING LIBRARIES------------
import pandas as pd
import numpy as np
import random
import seaborn as sns
from sklearn import preprocessing

sales_data = pd.read_csv('sales.csv')
sales_data

#--------------------- CLEANING THE DATA--------------------------

# dropping "Unnamed" for too many unique values and questionable purpose, "date" and "open" for lack of purpose for the algorythm (the "weekday" column information suffices)
sales_data.info
sales_data['Unnamed: 0'].unique()
sales_data.drop(columns=['Unnamed: 0', 'date', 'open'], inplace=True)


#checking if there is a difference in sales between the different classes of holidays i.e. if we should just make two groups (holiday vs. non-holiday) or keep 3 different classes of holidays
sales_data.groupby(by='state_holiday').agg({'sales':'sum'})

# replacing the letters in "state holiday" with numbers
sales_data['state_holiday'].unique()
sales_data['state_holiday'] = np.where(sales_data['state_holiday'] == 'a', 1 , sales_data['state_holiday'])
sales_data['state_holiday'] = np.where(sales_data['state_holiday'] == 'b', 2 , sales_data['state_holiday'])
sales_data['state_holiday'] = np.where(sales_data['state_holiday'] == 'c', 3 , sales_data['state_holiday'])
sales_data['state_holiday'] = np.where(sales_data['state_holiday'] == '0', 0 , sales_data['state_holiday'])
sales_data['state_holiday'] = sales_data['state_holiday'].astype(int)

# checking if any additional columns need to be dropped for high correlation
sales_data.corr()

# checking if standardization of data is necessary
ax = sns.boxplot(y="nb_customers_on_day", data=sales_data, palette="Set3")
# it is necessary since we have lots of outliers in a very decisive column

#standardizing all columns but the target column
sales_data_standardize = pd.DataFrame(preprocessing.StandardScaler().fit_transform(sales_data.drop(columns='sales')), columns=['store_ID', 'day_of_week', 'nb_customers_on_day', 'promotion','state_holiday', 'school_holiday'])
sales_data_standardize['sales']=sales_data['sales']

#----------TRAIN TEST SPLIT--------------
features = list(sales_data_standardize.columns)
features.remove('sales')

from sklearn.model_selection import train_test_split
y = sales_data_standardize['sales']
X = sales_data_standardize[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

# ------------TRYING OUT VARIOUS ENSEMBLE REGRESSORS-------------
# DECISION TREE
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=1)
tree.fit(X_train, y_train)
tree.score(X_test,y_test)


# BAGGING
from sklearn.ensemble import BaggingRegressor

bagging_reg = BaggingRegressor(
    DecisionTreeRegressor(max_depth=3),
    n_estimators=10,
    max_samples=100,
    random_state=1)

bagging_reg.fit(X_train, y_train)
bagging_reg.score(X_test,y_test)


# RANDOM FORREST
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=10,
                               max_depth=3,
                               random_state=1)
forest.fit(X_train, y_train)
forest.score(X_test,y_test)


# ADA BOOST
from sklearn.ensemble import AdaBoostRegressor

ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                            n_estimators=10,
                            random_state=1
                            )
ada_reg.fit(X_train, y_train)
ada_reg.score(X_test,y_test)


# GRADIENT BOOST
from sklearn.ensemble import GradientBoostingRegressor

gb_reg = GradientBoostingRegressor(max_depth=5,
                                   n_estimators=100,
                                   random_state=1
                                   )
gb_reg.fit(X_train, y_train)
gb_reg.score(X_test,y_test)


# XG BOOST
import xgboost
xgb_reg = xgboost.XGBRegressor(max_depth=13, n_estimators=550, n_jobs=10, random_state=0)
xgb_reg.fit(X_train, y_train)
xgb_reg.score(X_test,y_test)


#--------------HYPERPARAMETER TUNING------------

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Checking the accuracy for different max_depth
training_accuracy = []
test_accuracy = []
max_depth = range(10, 20)
for depth in max_depth:
    xgb_reg = xgboost.XGBRegressor(max_depth=depth, n_estimators = 550 ,random_state=0)
    xgb_reg.fit(X_train, y_train)
    training_accuracy.append(xgb_reg.score(X_train, y_train))
    test_accuracy.append(xgb_reg.score(X_test, y_test))

plt.plot(max_depth, training_accuracy, label="training accuracy")
plt.plot(max_depth, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("max_depth")
plt.show()


# First Grid Search
n_estimators = [10,100,500,1000]
max_depth = [5,10]

grid = {'n_estimators': n_estimators,
        'n_jobs':10,
        'max_depth': max_depth}
xgb_reg = xgboost.XGBRegressor()
grid_search = GridSearchCV(estimator = xgb_reg, param_grid = grid, cv = 5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.score(X_test, y_test))


# Second Grid Search (Adjusted Parameters)
n_estimators = [400,450,500,550,600]
max_depth = [10,11,12,13,14,15,16]

grid = {'n_estimators': n_estimators,
        'n_jobs':10,
        'max_depth': max_depth}
xgb_reg = xgboost.XGBRegressor()
grid_search = GridSearchCV(estimator = xgb_reg, param_grid = grid, cv = 5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.score(X_test, y_test))

# Checking accuracy for narrowed down n_estimators
training_accuracy = []
test_accuracy = []
n_estimators = range(520,540)
for est in n_estimators:
    xgb_reg = xgboost.XGBRegressor(n_estimators=est)
    xgb_reg.fit(X_train, y_train)
    training_accuracy.append(xgb_reg.score(X_train, y_train))
    test_accuracy.append(xgb_reg.score(X_test, y_test))

plt.plot(n_estimators, training_accuracy, label="training accuracy")
plt.plot(n_estimators, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_estimators")
plt.show()

#-----------------SAVING THE BEST MODEL TO A PICKLE-----------
import pickle
pickle.dump(xgb_reg, open('model2.p', 'wb'))
