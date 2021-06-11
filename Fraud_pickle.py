#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[150]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] =  14, 8
RANDOM_SEED = 42
LABELS = ["normal","fraud"]


# # import dataset

# In[151]:


data = pd.read_csv('sample.csv', sep=',')
data.head()


# In[ ]:





# # EDA

# In[152]:


data.info()


# In[153]:


data.isnull().values.any()


# In[154]:


#replace NaN values with 0
data = data.fillna(0)


# In[155]:


del data['x5'] 
del data['x6']
del data['x20']
del data['x27']
del data['x57']


# In[156]:


data.isnull().values.any()


# In[157]:


count_classes = pd.value_counts(data['y'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("y as class type")

plt.ylabel("Frequency")


# In[158]:


## Classify into  normal / fraud dataset

fraud = data[data['y']==1]

normal = data[data['y']==0]


# In[159]:


print(fraud.shape,normal.shape)


# In[160]:


fraud.x1.describe()
normal.x1.describe()


# In[161]:


f, (ax1, ax2) = plt.subplots(2,1, sharex = True)
f.suptitle('Amount per transaction  by column = y(class)')
bins = 50
ax1.hist(fraud.x90, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.x90, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('No. of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[162]:


outlier_fraction = len(fraud)/len(normal)


# In[163]:


print(outlier_fraction)

print("Fraud cases: {}".format(len(fraud)))
print("Normal cases: {}".format(len(normal)))


# In[164]:


#correlation

import seaborn as sns
#gte correlations of each features in dataset

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

 #plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[165]:


## take some sample of the data

data1= data.sample(frac = 0.1, random_state=1)

data1.shape


# In[166]:


#correlation

import seaborn as sns
#gte correlations of each features in dataset

corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

 #plot heat map
g=sns.heatmap(data1[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[167]:


#create independent / dependent features

columns =data.columns.tolist()
#filter the columns to remove data we dont want
columns = [c for c in columns if c not in ["y"]]
#store the variable we are perdicting
target = "y"

#define a random state
state = np.random.RandomState(42)
X=data[columns]
Y=data[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
#print the shapes of X & Y
print(X.shape)
print(Y.shape)


# In[ ]:





# In[ ]:





# In[174]:


#define the outlier detection methods

classifiers = {
       "Isolation Forest": IsolationForest(n_estimators =100, max_samples=len(X),
                                            contamination=outlier_fraction, random_state=state, verbose=0),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm='auto',
                                                  leaf_size=30, metric='minkowski',
                                                  p=2, metric_params=None, contamination=outlier_fraction),
         "Support Vector Machines":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05,
                                                   max_iter=-1)
}


# In[175]:


type(classifiers)


# In[176]:


n_outliers = len(fraud)
for i,(clf_name,clf) in enumerate(classifiers.items()):
    #fit the data and tag outliers
    if clf_name =="Local Outlier Factor":
        y_pred=clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support vector Machines":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:
        clf.fit(X)
        scores_predictions = clf.decision_function(X)
        y_pred = clf.predict(X)
        
#reshape the prediction values to 0 for valid transactions, 1- fraud trans

        y_pred[y_pred ==1] =0
        y_pred[y_pred ==-1] =1
        n_errors =(y_pred != Y).sum()
    
    
    #run classification metrics
    print("{}: {}". format(clf_name, n_errors))
    print("accuracy score:")
    print(accuracy_score(Y,y_pred))
    print("classification report:")
    print(classification_report(Y,y_pred))


# In[191]:


# #.pkl file as dump

import pickle as pkl


# In[ ]:





# In[192]:


#with open('chal_fraud_detect','wb') as f:
#    pickle.dump(classifiers, f)
file_dump = "untitled.pkl"


# In[193]:


file =  open(file_dump,'wb')
pkl.dump(classifiers, file)
file.close()


# In[ ]:





# In[ ]:





# In[182]:


#joblib
import joblib


# In[183]:


#joblib.dump(classifiers, 'classifier_joblib')


# In[ ]:





# In[185]:


#mj = joblib.load('classifier_joblib')


# In[186]:


#mj


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




