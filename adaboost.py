# Implementation of adaboost algorithm for classification

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)

# define the model
model = AdaBoostClassifier()

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# In this case, we can see the AdaBoost ensemble with default hyperparameters achieves a
# classification accuracy of about 80 percent on this test dataset.
# 1 Accuracy: 0.806 (0.041)
# We can also use the AdaBoost model as a final model and make predictions for classification.

# FOR PREDICTION
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [[0.20543991, -0.97049844, -0.81403429, -0.23842689, -0.60704084,
-0.48541492, 0.53113006, 2.01834338, -0.90745243, -1.85859731, -1.02334791,
-0.6877744, 0.60984819, -0.70630121, -1.29161497, 1.32385441, 1.42150747,
1.26567231, 2.56569098, -0.11154792]]

yhat = model.predict(row)
print('Predicted Class: %d' % yhat[0])
