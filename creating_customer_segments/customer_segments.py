# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
import matplotlib as plt

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    
# Display a description of the dataset
display(data.describe())

# TODO: Select three indices of your choice you wish to sample from the dataset
#indices = [45,76,99]

# Create a DataFrame of the chosen samples
#samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
#print "Chosen samples of wholesale customers dataset:"
#display(samples)

from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

#print data
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature

label = data['Fresh']
data.drop(['Fresh'], axis = 1, inplace = True)

#new_label = data['Fresh']

# TODO: Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(data,label,test_size=0.25, random_state=10)


from sklearn.tree import DecisionTreeRegressor

# TODO: Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=0);
regressor.fit(X_train,y_train);
y_pred = regressor.predict(X_test)
print regressor.score(X_test,y_test)


# TODO: Report the score of the prediction using the testing set
#score = f1_score(y_test, y_pred)


