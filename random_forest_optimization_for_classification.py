# Explore Number of Samples
# Each decision tree in the ensemble is fit on a bootstrap sample drawn from the training dataset.
# This can be turned off by setting the “ bootstrap” argument to False, if you desire.
# In that case, the whole training dataset will be used to train each decision tree. This is not recommended.
# The “max_samples” argument can be set to a float between 0 and 1 to control the percentage of the
# size of the training dataset to make the bootstrap sample used to train each decision tree.
# For example, if the training dataset has 100 rows, the max_samples argument could be set to 0.5 and
# each decision tree will be fit on a bootstrap sample with (100 * 0.5) or 50 rows of data.
# A smaller sample size will make trees more different, and a larger sample size will make the trees more similar.
# Setting max_samples to “None” will make the sample size the same size as the training dataset
# and this is the default. The example below demonstrates the effect of different bootstrap
# sample sizes from 10 percent to 100 percent on the random forest algorithm.

# explore random forest bootstrap sample size on performance
from numpy import mean
from numpy import std
from numpy import arange
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
    return X, y

# get a list of models to evaluate
def get_models():
    models = dict()
    # explore ratios from 10% to 100% in 10% increments
    for i in arange(0.1, 1.0, 0.1):
        key = '%.1f' % i
        # set max_samples=None to use 100%
        if i == 1.0:
            i = None
        models[key] = RandomForestClassifier(max_samples=i)
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

# Define dataset
X, y = get_dataset()
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X, y)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

# Note: Your results may vary given the stochastic nature of the algorithm or evaluation procedure,
# or differences in numerical precision.
# Consider running the example a few times and compare the average outcome.
