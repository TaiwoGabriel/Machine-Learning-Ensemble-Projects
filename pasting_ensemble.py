# The Pasting Ensemble is an extension to bagging that involves fitting ensemble members
# based on random samples of the training dataset instead of bootstrap samples.
# The approach is designed to use smaller sample sizes than the training dataset in cases
# where the training dataset does not fit into memory.

# The example below demonstrates the Pasting ensemble by setting the “ bootstrap” argument
# to “False” and setting the number of samples used in the training dataset via “max_samples”
# to a modest value, in this case, 50 percent of the training dataset size.

# evaluate pasting ensemble algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# define the model
model = BaggingClassifier(bootstrap=False, max_samples=0.5)
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
