import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV


def evalerror(preds, dtrain):
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(dtrain))

def modelFit(alg, X, y,X_test, y_test,useTrainCV=True, cvFolds=5, early_stopping_rounds=50):
    if useTrainCV:
        print "cvFolds", cvFolds
        xgbParams = alg.get_xgb_params()
        #xgbParams['num_class']=300
        xgTrain = xgb.DMatrix(X, label=y)
        #cvresult = xgb.cv(xgbParams,
        #              xgTrain,
        #              num_boost_round=alg.get_params()['n_estimators'],
        #              nfold=cvFolds,
        #              stratified=False,
        #              metrics={'mae'},
        #              early_stopping_rounds=early_stopping_rounds,
        #              seed=0)

        #print cvresult
        #alg.set_params(n_estimators=cvresult.shape[0])

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
X = train.drop(['loss', 'id'], 1)
X_test = test.drop(['loss', 'id'], 1)


param_test1={'max_depth':range(3,10,2),'min_child_weight':range(1,8,2)}

xgb2 = XGBRegressor(learning_rate =0.01,
                    n_estimators=741,
                    max_depth=12,
                    min_child_weight=1,
                    gamma=1,
                    subsample=0.8,
                    colsample_bytree=0.5,
                    scale_pos_weight=1,
                    objective='reg:linear',
                    seed=2016)

gsearch1=GridSearchCV(estimator=xgb2,param_grid=param_test1,scoring='mean_absolute_error',iid=False,cv=5)
gsearch1.fit(X, y,)
print gsearch1.grid_scores_,"\n"
print gsearch1.best_params_,"\n"
print gsearch1.best_score_,"\n"

clf=gsearch1.best_estimator_

y_test=[]
modelFit(clf, X, y, X_test, y_test)
pickle.dump(clf,open("2_opt_child.pkl","wb"))

clf2=pickle.load(open("2_opt_child.pkl","rb"))
prediction = clf2.predict(X) 
print "mean_absolute_error (Train): %f", evalerror(prediction,y)


prediction_test = clf2.predict(X_test) 
submission = pd.DataFrame()
submission['loss'] = prediction
submission['id'] = ids
submission.to_csv('sub_2_opt_num_child_size.csv', index=False)

