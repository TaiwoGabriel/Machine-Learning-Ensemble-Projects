# Import Libraries

from numpy import mean,std
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from mlxtend.evaluate import bias_variance_decomp
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,VotingClassifier
from sklearn.ensemble import StackingClassifier

# Importing dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
cv_method = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

# Hyperparameter Optimization using RandomSearch and CrossValidation to get the best model hyperparamters

# naive Bayes Classifier
NB = GaussianNB()
# Create a dictionary of naive bayes parameters
params_NB = {'var_smoothing': np.logspace(0,-9,num=100) }
# var_smoothing indicates the laplace correction
# Also, priors represents the prior probabilities of the classes. If we specify this parameter while
# fitting the data, then the prior probabilities will not be justified according to the data
# Computing the RandomSearch
NB_grid = RandomizedSearchCV(NB,params_NB,scoring='accuracy',cv=cv_method)
# Fitting the NB_grid
NB_grid.fit(X_train,y_train)
# Print the best parameter values
print(' NB Best Parameter Values:', NB_grid.best_params_)
GNB = GaussianNB(**NB_grid.best_params_)


# kNN Classifier
nearest_neighbour = KNeighborsClassifier()
# Create a dictionary of KNN parameters
# K values between 1 and 9 are used to avoid ties and p values of 1 (Manhattan), 2 (Euclidean), and 5 (Minkowski)
param_kNN = {'n_neighbors': [1,3,5,7,9],'p':[1,2,5]} # Distance Metric: Manhattan (p=1), Euclidean (p=2) or
# Minkowski (any p larger than 2). Technically p=1 and p=2 are also Minkowski distances.
# Define the kNN model using RandomSearch and optimize accuracy
kNN_grid = RandomizedSearchCV(nearest_neighbour,param_kNN,scoring='accuracy',cv=cv_method)
kNN_grid.fit(X_train,y_train)
# Print the best parameter values for KNN
print('kNN Best Parameter values =',kNN_grid.best_params_)
kNN = KNeighborsClassifier(**kNN_grid.best_params_)

# Decision Tree Classifier
Decision_Tree = DecisionTreeClassifier()
# Create a dictionary of DT hyperparameters
params_DT = {'criterion':['gini','entropy'],
             'max_depth':[1,2,3,4,5,6,7,8],
             'splitter':['best','random']}

# Using Random Search to explore the best parameter for the a decision tree model
DT_Grid = RandomizedSearchCV(Decision_Tree,params_DT,scoring='accuracy',cv=cv_method)
# Fitting the parameterized model
DT_Grid.fit(X_train,y_train)
# Print the best parameter values
print('DT Best Parameter Values:', DT_Grid.best_params_)

DT = DecisionTreeClassifier(**DT_Grid.best_params_)


# Support Vector Machines
SVM_clasf = SVC(probability=True)
# Create a dictionary of SVM hyperparameters
# Parameter space for linear and rbf kernels
params_SVC = {'kernel':['rbf'],'C':np.linspace(0.1,1.0),
              'gamma':['scale','auto']} #np.linspace(0.1,1.0)}
# Using Random Search to explore the best parameter for the a SVM model
SVC_Grid = RandomizedSearchCV(SVM_clasf,params_SVC,scoring='accuracy',cv=cv_method)
# Fitting the parameterized model
SVC_Grid.fit(X_train,y_train)
# Print the best parameter values
print('SVC Best Parameter Values:', SVC_Grid.best_params_)
SVC = SVC(**SVC_Grid.best_params_)


# Neural Network
mlp = MLPClassifier(early_stopping=True)
parameter_MLP = {
    'hidden_layer_sizes': [(100,100,100),(150,150,150)],
    'activation': ['relu','tanh'],
    'solver': ['sgd'],'max_iter':[500, 1000],
    'learning_rate': ['constant','adaptive']}

mlp_Grid = RandomizedSearchCV(mlp, parameter_MLP, scoring='accuracy',cv=cv_method)
mlp_Grid.fit(X_train, y_train) # X is train samples and y is the corresponding labels

# Check best hyperparameters
print('ANN Best parameters found:\n', mlp_Grid.best_params_)
MLP = MLPClassifier(**mlp_Grid.best_params_)


# Develop Homogeneous and Heterogeneous models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('NB', GNB))
    level0.append(('kNN', kNN))
    level0.append(('C5.0', DT))
    level0.append(('SVM', SVC))
    level0.append(('ANN', MLP))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1,cv=10)
    return model

GNB_ensemble = BaggingClassifier(base_estimator=GNB,n_estimators=10)
kNN_ensemble = BaggingClassifier(base_estimator=kNN,n_estimators=10)
DT_ensemble = BaggingClassifier(base_estimator=DT,n_estimators=10)
Rand_Forest = RandomForestClassifier(n_estimators=10)
SVM_ensemble = BaggingClassifier(base_estimator=SVC,n_estimators=10)
MLP_ensemble = BaggingClassifier(base_estimator=MLP,n_estimators=10)


# get a list of models to evaluate
def get_models():
    models = dict()
    models['NB_HE'] = GNB_ensemble
    models['kNN_HE'] = kNN_ensemble
    models['DT_HE'] = DT_ensemble
    models['SVM_HE'] = SVM_ensemble
    models['ANN_HE'] = MLP_ensemble
    models['Stacking'] = get_stacking()
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv_method, n_jobs=-1)
    return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

print()
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()


