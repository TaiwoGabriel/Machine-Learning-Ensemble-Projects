
# This is an alternate approach to implement gradient tree boosting inspired by the
# LightGBM library (described more later). This implementation is provided via the
# HistGradientBoostingClassifier and HistGradientBoostingRegressor classes.
# The primary benefit of the histogram-based approach to gradient boosting is speed.
# These implementations are designed to be much faster to fit on training data.

# The example below first evaluates a HistGradientBoostingClassifier on the test problem
# using repeated k-fold cross-validation and reports the mean accuracy. Then a single
# model is fit on all available data and a single prediction is made

# histogram-based gradient boosting for classification in scikit-learn
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
n_redundant=5, random_state=1)

# evaluate the model
model = HistGradientBoostingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1,
error_score='raise')
# print('Accuracy: ', mean(n_scores), '\u00B1', std(n_scores))
#print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
print('Mean Accuracy and Std Dev: %.3f' %  (mean(n_scores)), '\u00B1',  ' %.3f' % std(n_scores))

# fit the model on the whole dataset
model = HistGradientBoostingClassifier()
model.fit(X, y)
# make a single prediction
row = [[2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951,
-1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]]
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])
