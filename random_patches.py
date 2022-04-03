# The Random Patches Ensemble is an extension to bagging that involves fitting ensemble
# members based on datasets constructed from random subsets of rows (samples) and columns (features)
# of the training dataset.
# It does not use bootstrap samples and might be considered an ensemble that combines both the
# random sampling of the dataset of the Pasting ensemble and the random sampling of features of
# the Random Subspace ensemble.

# The example below demonstrates the Random Patches ensemble with decision trees created
# from a random sample of the training dataset limited to 50 percent of the size of the
# training dataset, and with a random subset of 10 features.

# evaluate random patches ensemble algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# define the model
model = BaggingClassifier(bootstrap=False, max_features=10, max_samples=0.5)
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
