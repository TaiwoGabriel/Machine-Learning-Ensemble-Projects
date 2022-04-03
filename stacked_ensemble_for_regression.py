# Specifically, we will evaluate the following three algorithms:
# k-Nearest Neighbors.
# Decision Tree.
# Support Vector Regression.
# Note: The test dataset can be trivially solved using a linear regression model as the
# dataset was created using a linear model under the covers. # As such, we will leave this model out of
# the example so we can demonstrate the benefit of the stacking ensemble method.
# Each algorithm will be evaluated using the default model hyperparameters.
# Each model will be evaluated using repeated k-fold cross-validation
# We can then report the mean performance of each algorithm and also create a box and whisker
# plot to compare the distribution of accuracy scores for each algorithm.
# In this case, model performance will be reported using the mean absolute error (MAE).
# The scikit-learn library inverts the sign on this error to make it maximizing, from -infinity to 0 for
# the best score.

# compare machine learning models for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from matplotlib import pyplot

# get the dataset
def get_dataset():
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
    return X, y

# get a list of models to evaluate
def get_models():
    models = dict()
    models['knn'] = KNeighborsRegressor()
    models['cart'] = DecisionTreeRegressor()
    models['svm'] = SVR()
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
print()

# Here we have three different algorithms that perform well, presumably in different ways on this dataset.
# Next, we can try to combine these three models into a single ensemble model using stacking.
# We can use a linear regression model to learn how to best combine the predictions from each of
# the separate three models.
# Our expectation is that the stacking ensemble will perform better than any single base model.
# This is not always the case, and if it is not the case, then the base model should be used in
# favor of the ensemble model.

# compare ensemble to each standalone models for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from matplotlib import pyplot

# get the dataset
def get_dataset():
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
    return X, y

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('cart', DecisionTreeRegressor()))
    level0.append(('svm', SVR()))
    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model

# get a list of models to evaluate
def get_models():
    models = dict()
    models['knn'] = KNeighborsRegressor()
    models['cart'] = DecisionTreeRegressor()
    models['svm'] = SVR()
    models['stacking'] = get_stacking()
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
print()
# In this case, we can see that the stacking ensemble appears to perform better than any single
# model on average, achieving a mean negative MAE of about -56.
# A box plot is created showing the distribution of model error scores.
# Here, we can see that the mean and median scores for the stacking model sit
# much higher than any individual model.

# If we choose a stacking ensemble as our final model, we can fit and use it to make
# predictions on new data just like any other model.
# First, the stacking ensemble is fit on all available data, then the predict() function can
# be called to make predictions on new data.
# The example below demonstrates this on our regression dataset.

# make a prediction with a stacking ensemble
# fit the model on all available data
model = get_stacking()
model.fit(X, y)
# make a prediction for one example
data = [[5.88891819,2.64867662,-0.42728226,-1.24988856,-0.00822,-3.57895574,2.87938412,
        -1.55614691,-0.38168784,7.50285659,-1.16710354,-5.02492712,3.23098,
        -2.908754, -1.67432, 2.1093543, 1.324189, 0.654219,4.23160,-3.1023174]]

yhat = model.predict(data)
print('Predicted Value: %.3f' % (yhat))
