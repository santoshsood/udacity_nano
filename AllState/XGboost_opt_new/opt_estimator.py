
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import pickle

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error

def evalerror(preds, dtrain):
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(dtrain))

def modelFit(alg, X, y,X_test, y_test,useTrainCV=True, cvFolds=5, early_stopping_rounds=50):
    if useTrainCV:
        print "cvFolds", cvFolds
        xgbParams = alg.get_xgb_params()
        #xgbParams['num_class']=300
        xgTrain = xgb.DMatrix(X, label=y)
        cvresult = xgb.cv(xgbParams,
                      xgTrain,
                      num_boost_round=alg.get_params()['n_estimators'],
                      nfold=cvFolds,
                      stratified=False,
                      metrics={'mae'},
                      early_stopping_rounds=early_stopping_rounds,
                      seed=0)

        print cvresult
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm
    alg.fit(X, y, eval_metric='mlogloss')
    # Predict
    y_val_pred = alg.predict(X)
    y_test = alg.predict(X_test)
    #y_prob_pred = alg.predict_proba(X)

    # Print model report:
    print "\nModel Report"
    print "Classification report: \n"
    print "y_val",y
    print "y_val_pred",y_val_pred
    print "mean_absolute_error (Train): %f", evalerror(y_val_pred,y.values)
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    #plt.show()


# 1) Read training set
print('>> Read training set')
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])
print train.shape
print test.shape
print joined.shape


for column in list(train.select_dtypes(include=['object']).columns):
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)
            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x
            joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
        joined[column] = pd.factorize(joined[column].values, sort=True)[0]
 
train = joined[joined['loss'].notnull()]
test = joined[joined['loss'].isnull()]
print train.shape
print test.shape
print joined.shape

# 2) Extract target attribute and convert to numeric
shift = 200
y = np.log(train['loss'] + shift)

ids = test['id']
print ids
X = train.drop(['loss', 'id'], 1)
X_test = test.drop(['loss', 'id'], 1)

# 5) First classifier
xgb1 = XGBRegressor(learning_rate =0.01,
                    n_estimators=2000,
                    max_depth=12,
                    min_child_weight=1,
                    gamma=1,
                    subsample=0.8,
                    colsample_bytree=0.5,
                    scale_pos_weight=1,
                    objective='reg:linear',
                    seed=2016)
y_test=[]
modelFit(xgb1, X, y, X_test, y_test)

xgtest = xgb.DMatrix(X_test)
xgtrain = xgb.DMatrix(X)
pickle.dump(xgb1,open("opt_est.pkl","wb"))

clf2=pickle.load(open("opt_est.pkl","rb"))
prediction = clf2.predict(X) 
print "mean_absolute_error (Train): %f", evalerror(prediction,y)


prediction_test = clf2.predict(X_test) 
submission = pd.DataFrame()
submission['loss'] = prediction
submission['id'] = ids
submission.to_csv('sub_opt_est.csv', index=False)



