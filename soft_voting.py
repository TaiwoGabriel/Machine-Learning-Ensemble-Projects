# Soft Voting. Predict the class with the largest summed probability from models.

# We can demonstrate soft voting with the support vector machine (SVM) algorithm.
# The SVM algorithm does not natively predict probabilities, although it can be configured to predict
# probability-like scores by setting the “probability” argument to “True” in the SVC class.
# We can fit five different versions of the SVM algorithm with a polynomial kernel,
# each with a different polynomial degree, set via the “ degree” argument. We will use degrees 1-5.
# Our expectation is that by combining the predicted class membership probability scores
# predicted by each different SVM model that the soft voting ensemble will achieve a better
# predictive performance than any standalone model used in the ensemble, on average.

# compare soft voting ensemble to standalone classifiers
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
    return X, y

# get a voting ensemble of models
def get_voting():
    # define the base models
    models = list()
    models.append(('svm1', SVC(probability=True, kernel='poly', degree=1)))
    models.append(('svm2', SVC(probability=True, kernel='poly', degree=2)))
    models.append(('svm3', SVC(probability=True, kernel='poly', degree=3)))
    models.append(('svm4', SVC(probability=True, kernel='poly', degree=4)))
    models.append(('svm5', SVC(probability=True, kernel='poly', degree=5)))
    # define the voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='soft')
    return ensemble

# get a list of models to evaluate
def get_models():
    models = dict()
    models['svm1'] = SVC(probability=True, kernel='poly', degree=1)
    models['svm2'] = SVC(probability=True, kernel='poly', degree=2)
    models['svm3'] = SVC(probability=True, kernel='poly', degree=3)
    models['svm4'] = SVC(probability=True, kernel='poly', degree=4)
    models['svm5'] = SVC(probability=True, kernel='poly', degree=5)
    models['soft_voting'] = get_voting()
    return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
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

# If we choose a soft voting ensemble as our final model, we can fit and use it to make
# predictions on new data just like any other model. First, the soft voting ensemble is fit on all
# available data, then the predict() function can be called to make predictions on new data.
# The example below demonstrates this on our binary classification dataset.

# make a prediction with a soft voting ensemble
# fit the model on all available data
ensemble = get_voting()
ensemble.fit(X, y)
# make a prediction for one example
data = [[5.88891819,2.64867662,-0.42728226,-1.24988856,-0.00822,-3.57895574,2.87938412,
        -1.55614691,-0.38168784,7.50285659,-1.16710354,-5.02492712,3.23098,
        -2.908754, -1.67432, 2.1093543, 1.324189, 0.654219,4.23160,-3.1023174]]
yhat = ensemble.predict(data)
print('Predicted Class: %d' % (yhat))
