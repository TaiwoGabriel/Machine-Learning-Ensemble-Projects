# we will evaluate the model using repeated k-fold cross-validation, with three repeats and 10 folds.
# We will report the mean absolute error (MAE) of the model across all repeats and folds.
# The scikit-learn library makes the MAE negative so that it is maximized
# instead of minimized. This means that larger negative MAE are better and a perfect model has a MAE of 0.

# Implementation of bagging ensemble for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import BaggingRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=5)
# define the model
model = BaggingRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [[5.88891819,2.64867662,-0.42728226,-1.24988856,-0.00822,-3.57895574,2.87938412,
        -1.55614691,-0.38168784,7.50285659,-1.16710354,-5.02492712,3.23098,
        -2.908754, -1.67432, 2.1093543, 1.324189, 0.654219,4.23160,-3.1023174]]
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])
