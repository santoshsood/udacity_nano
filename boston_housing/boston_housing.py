import numpy as np
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer


# Pretty display for notebooks
import matplotlib as plot

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
prices=np.array(prices);

# TODO: Minimum price of the data
minimum_price = np.nanmin(prices);

# TODO: Maximum price of the data
maximum_price = np.nanmax(prices);

# TODO: Mean price of the data
mean_price = np.nanmean(prices);

# TODO: Median price of the data
median_price = np.nanmedian(prices);

# TODO: Standard deviation of prices of the data
std_price = np.nanstd(prices);

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = mean_squared_error(y_true,y_predict)**0.5;
    
    # Return the score
    return score
# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)

# TODO: Import 'train_test_split'

# TODO: Shuffle and split the data into training and testing subsets
N=len(prices);
#print"N",N;
X_train, X_test, y_train, y_test = (None, None, None, None)

sample=100;
train_idx = np.unique(np.random.choice(N, sample));
train_idx=np.array(train_idx);
#print"train_idx", train_idx;

test_idx = [idx for idx in xrange(N) if idx not in train_idx];
test_idx=np.array(test_idx);
#print"test_idx", test_idx;

#print "Size train_idx test_idx", len(train_idx),len(test_idx);


Y_train=prices[train_idx];
Y_test=prices[test_idx];

features=np.array(features);

X_train=features[train_idx];
X_test=features[test_idx];

#print "features", features;
#print "Size features", len(features);
#print "X_train", X_train;
#print "X_test", X_test;
#print "Size X_train", len(X_train), "X_test", len(X_test);
#print "Size Y_train", len(Y_train) ,"Y_test", len(Y_test);
#print "Size features", len(features);

#test_idx = [idx for idx in xrange(N) if idx not in train_idx]
#print"test_idx", train_idx;
# test_idx = np.random.choice(N, sample)
#X_test = X[:test_idx]
#Y_test = Y[test_idx]


# Success
print "Training and testing split was successful."

# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'


def fit_model(X,Y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    
    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(max_leaf_nodes=30, random_state=0);
    
    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': [1,2,3,4,5,6,7,8,9,10]}
    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric,greater_is_better="TRUE");

    # TODO: Create the grid search object
    grid =  GridSearchCV(estimator=regressor,param_grid=params, cv=cv_sets);

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X,Y)
    print "grid  \n", grid.grid_scores_ ;
    

    # Return the opthimal model after fitting the data
    return grid.best_estimator_



# Fit the training data to the model using grid search
reg = fit_model(X_train, Y_train);

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])



