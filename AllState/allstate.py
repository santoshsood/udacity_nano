# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.metrics import f1_score
from pandas.tools.plotting import scatter_matrix
from sklearn import preprocessing
from IPython.display import display
from statsmodels.graphics.mosaicplot import mosaic


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "./"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
train = pd.read_csv("./train.csv")
train.describe(include = ['object'])
display(train.head(5))

loss = train['loss']
features=train.drop('loss',axis=1)
print ("dataset has {} data points with {} variables each.".format(*train.shape))

display(loss.describe())
#display(features.describe())
#observation max loss is hugh outlier


for i in train.columns:
    if (train[i].isnull().sum() > 0):
      print("missing data in feature {}", i)


cont_columns = []
cat_columns = []

for i in train.columns:
    if train[i].dtype == 'float':
        cont_columns.append(i)
    elif train[i].dtype == 'object':
        cat_columns.append(i)
print cont_columns
print cat_columns


plt.figure()
sns.pairplot(train[cont_columns], vars=cont_columns, kind = 'scatter',diag_kind='kde')
plt.show()
        
print cont_columns[1:10]
#sns.jointplot(x='cont1',y='cont2',data=train)
pd.scatter_matrix(train[cont_columns], alpha = 0.3, figsize = (14,8), diagonal = 'kde');
plt.show()

#for cat in cat_columns:
#    sns.stripplot(x=cat,y='loss', data=train)
#    plt.show()
print cont_columns[1:10]


       