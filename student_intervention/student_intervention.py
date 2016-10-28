# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import f1_score
from pandas.tools.plotting import scatter_matrix
from sklearn import preprocessing
from IPython.display import display
from statsmodels.graphics.mosaicplot import mosaic

# Read student data
student_data = pd.read_csv("student-data.csv")
#print(student_data.head(2))


def handle_non_numeric_data(df):
    columns=df.columns.values
    for column in columns:
       if (df[column].dtype != np.int64 or df[column].dtype != np.float):
        le = preprocessing.LabelEncoder()
        le.fit(df[column])
        list(le.classes_)
        df[column]=le.transform(df[column])
        #print "Column", column, "List", list(le.classes_)

    return

#display(student_data)

print "Student data read successfully!"

# read data in plot_student_data for plotting in 


feature_cols = list(student_data.columns[:-1])
print feature_cols
plt.rcParams['font.size'] = 8.0


for i, feature in enumerate(feature_cols):
    print i ,feature
    mosaic(student_data, [feature,'passed']);
    #student_data.groupby(feature).size().plot(kind='bar')
    plt.xlabel(feature)
    #plt.show()

 #print student_data.head(2)
#features=student_data.drop(['passed'],axis=1)
#result=student_data['passed']

#print features.head(2)
#print result.shape[0] 

# TODO: Calculate number of students
n_students = student_data.shape[0] 

# TODO: Calculate number of features
#n_features = features.shape[1]
n_features = student_data.shape[1] -1


# TODO: Calculate passing students
n_passed = n_students *student_data['passed'].value_counts('yes')[0]

# TODO: Calculate failing students
n_failed = n_students*student_data['passed'].value_counts('no')[1]

# TODO: Calculate graduation rate
grad_rate = student_data['passed'].value_counts('yes')[0]*100.0

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head(5)

print "\nTarget values:"
print y_all.head(5)

#le = preprocessing.LabelEncoder()
#le.fit(y_all)
#y_all=le.transform(y_all)


#X_all_copy=X_all

##Print features & classes of each feature
#for feature_type in feature_cols:
# le = preprocessing.LabelEncoder()
# le.fit(X_all_copy[feature_type])
# X_all_copy[feature_type]=le.transform(X_all_copy[feature_type])   
# print feature_type, list(le.classes_)
   
 
 
# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))
y_all = y_all.replace(['yes', 'no'], [1, 0]) # changed to enable f1 scoring during grid search

#feature_col=list(X_all.columns[:])
#print feature_col 
#for i, feature in enumerate(feature_col):
#    print i ,feature

#feature_cols = list(X_all.columns[:-1])

##for feature_type in feature_cols:
# le = preprocessing.LabelEncoder()
# le.fit(X_all[feature_type])
# print feature_type, list(le.classes_)
# X=X_all[feature_type];
# print X 
# index_pass=
# plt.scatter(X, y_all, marker=">")
# plt.show()   
   
#le = preprocessing.LabelEncoder()
#le.fit(y_all)
#list(le.classes_)
#y_all=le.transform(y_all)
#print y_all

#print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split

# TODO: Set the number of training points
num_train = 0.8*X_all.shape[0]

# Set the number of testing points
num_test = X_all.shape[0] - num_train
test_size=num_test/(num_test+num_train)


X_train, X_test, y_train, y_test = train_test_split(X_all,y_all,test_size=test_size, random_state=10)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

#print "\n X-train", X_train.head(2)
#print "\n X-test", X_test.head(2)
#print "\n Y-train", y_train.head(2)
#print "\n Y-test", y_test.head(2)


from time import time
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydotplus

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    #print "\n target" ,target.values
    #print "\n prediction", y_pred
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label=1)


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    
    train_classifier(clf, X_train, y_train)
    if "Decision" in (clf.__class__.__name__):
     tree.export_graphviz(clf,out_file='tree.dot')    
     dot_data = StringIO() 
     tree.export_graphviz(clf, out_file=dot_data) 
     graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
     graph.write_pdf("ouput.pdf")
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    


# TODO: Import the three supervised learning models from sklearn
# from sklearn import model_A
# from sklearn import model_B
# from skearln import model_C
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

# TODO: Initialize the three models
clf_A = DecisionTreeClassifier();
clf_B = GaussianNB()
clf_C = svm.SVC()
    

X_train_100 = X_train[:100]
y_train_100 = y_train[:100]


X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size

# TODO: Choose a model, import it and instantiate an object
# Fit model to training data
train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
print" \n"
train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
print" \n"
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
print" \n"
train_predict(clf_A, X_train, y_train, X_test, y_test)

print" \n"
train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
print" \n"
train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
print" \n"
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
print" \n"
train_predict(clf_B, X_train, y_train, X_test, y_test)


print" \n"
train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
print" \n"
train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
print" \n"
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)
print" \n"
train_predict(clf_C, X_train, y_train, X_test, y_test)





# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics.scorer import make_scorer
from sklearn.cross_validation import ShuffleSplit

parameters = [{
    'max_depth': np.arange(1,30,1),
     'max_leaf_nodes' : np.arange(30,60,1)
      }]

# TODO: Initialize the classifier
clf = DecisionTreeClassifier()

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label=1)

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters, f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print clf
# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))


       
# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics.scorer import make_scorer
from sklearn.cross_validation import ShuffleSplit

parameters = [{
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
 }]

# TODO: Initialize the classifier
clf = svm.SVC()

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label=1)

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters, f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print clf

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))