
# Regression Example: Boston Housing Data
# Let's consider the Boston housing dataset. We call KNeighborsRegressor to run KNN on this regression problem.
# The KNN regression grid search is similar to its classification counterpart except for the differences below.
# We can no longer use stratified K-fold validation since the target is not multiclass or binary.
# However, we can use other methods such as K-fold or Repeated K-fold.
# The model performance metric is no longer "accuracy", but MSE (Mean Squared Error).
# We do not need to specify "mse" in GridSearchCV since sklearn is smart enough to
# figure out that the target is a continuous variable.

from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import  mean_squared_error, confusion_matrix, classification_report,plot_confusion_matrix
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score
from mlxtend.evaluate import bias_variance_decomp
from mlxtend.plotting import plot_learning_curves

# Calling data from the sklearn database
housing_data = load_boston()


# View the data on DataFrame
data = pd.DataFrame(housing_data['data'], columns=housing_data['feature_names'])
data['class'] = housing_data['target']
#print(data)

# Separate data from class.
X = housing_data.data
y = housing_data.target

# It can also be separated using the code below
#X = my_data[:,:-1] # X represents features
#Y = my_data[:,-1] # Y represents class labels


# Data normalization by scaling features and target
data_scale = MinMaxScaler()
X = data_scale.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross Validation
cv_method = RepeatedKFold(n_splits=5,n_repeats=3,random_state=42)

# Call kNN Regressor
kNN_Reg = KNeighborsRegressor()

# Hyperparameter optimization using GridSearch
params_kNN_Reg = {'n_neighbors': [1,2,3,4,5,6,7], 'p':[1,2,5]}

# Grid search regressor
GS_kNN_Reg = GridSearchCV(kNN_Reg,params_kNN_Reg,cv=cv_method,verbose=0) # Note: When verbose is set to 1, more
# info about the processing will be printed. But when verbose is set to 0, no info is printed.

# Fit the optimized kNN Regressor
GS_kNN_Reg.fit(X_train,y_train)

print('Best KNN Parameter:', GS_kNN_Reg.best_params_)

print('Mean of Cross Validation:', GS_kNN_Reg.best_score_)

# Evaluating the kNN Regression model on the test set using mean squared error and other metrics
y_pred = GS_kNN_Reg.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print('Mean squared score on test samples =',mse)
Reg_score = GS_kNN_Reg.score(X_test,y_test)
print('Regression score:', Reg_score)

# Plot kNN learning curve
plot_learning_curves(X_train,y_train,X_test,y_test, GS_kNN_Reg,scoring='misclassification error',style='ggplot',
                     print_model=True,legend_loc='best')
plt.title('kNN Regression Learning Curve')
plt.show()

# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(GS_kNN_Reg,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for kNN %.2f' % avg_expected_loss)
print('Average Expected Squared loss for Bias error for kNN %.2f' % avg_bias)
print('Average Expected Squared loss for Variance error for kNN %.2f' % avg_variance)
print()

avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(GS_kNN_Reg,X_train,y_train
                                                                 ,X_test,y_test,loss='0-1_loss',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected 0-1 loss for kNN %.2f' % avg_expected_loss2)
print('Average Expected 0-1 Bias error for kNN %.2f' % avg_bias2)
print('Average Expected 0-1 Variance error for kNN %.2f' % avg_variance2)






























