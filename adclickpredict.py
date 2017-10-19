import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from catboost import CatBoostClassifier


#Import train test

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



#--------------------------------------------------------------------Exploratory DataAnalysis-----------------------------------------------------------------

train.shape
train.corr()
train.isnull().sum()
train.sample()
train['devid'].value_counts()
train.hist()
plt.show()
train.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()
scatter_matrix(train)

#----------------------------------------------------------------Data Preparation-----------------------------------------------------------------------------
for col in train.columns:
    train[col] = train[col].astype('str')



#Correct Spelling mistakes and group same browsers spelt differently

train['browserid'] = [re.sub(pattern='^Mozilla$',repl='Mozilla Firefox',string=x) for x in train['browserid']]
train['browserid'] = [re.sub(pattern='^Firefox$',repl='Mozilla Firefox',string=x) for x in train['browserid']]
train['browserid'] = [re.sub(pattern='InternetExplorer',repl='IE',string=x) for x in train['browserid']]
train['browserid'] = [re.sub(pattern='Internet Explorer',repl='IE',string=x) for x in train['browserid']]
train['browserid'] = [re.sub(pattern='Google Chrome',repl='Chrome',string=x) for x in train['browserid']]


rows = np.random.choice(train.index.values, 2000000)


sampled_train = train.loc[rows]
train1= train.dropna()


#Fill missing values


#Fill site id based on merchant id

siteidmode={}
for merchantid in set(train['merchant']):
    siteidmode[merchantid]=train1[train1['merchant']==merchantid]['siteid'].value_counts()

#Fill browser based on country code

browseridmode={}
for countrycode in set(train['countrycode']):
    browseridmode[countrycode]=train1[train1['countrycode']==countrycode]['browserid'].mode()

#Fill device id based on browser

devidmode={}
for browserid in set(train['browserid']):
    devidmode[browserid] = train1[train1['browserid']==countrycode]['devid'].mode()

def expectedsiteid(row):
    if(row['siteid']!=row['siteid']):
        return siteidmode[row['merchant']]
    else:
        return row['siteid']

def expectedbrowser(row):
    if(row['browserid']!=row['browserid']):
        return browseridmode[row['countrycode']]
    else:
        return row['browserid']


def expecteddevid(row):
    if(row['devid']!=row['devid']):
        return devidmode[row['browserid']]
    else:
        return row['devid']

train = train.loc[rows]

train['siteid']=train.apply(expectedsiteid,axis=1)
train['browserid']= train.apply(expectedbrowser,axis=1)
train['devid']= train.apply(expecteddevid,axis=1)


train['datetime']= pd.to_datetime(train['datetime'])
train = pd.concat([train,pd.get_dummies(train['browserid'],prefix='browser'),pd.get_dummies(train['devid'],prefix='dev'),pd.get_dummies(train['countrycode'],prefix='country'),pd.get_dummies(train['category'],prefix='cat')],axis=1)

#Create seperate features for day, hour, minute

train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

#Treat Variables for cyclic nature

train['xweekday']= np.sin(2*3.14*train['tweekday']/31)
train['yweekday']= np.cos(2*3.14*train['tweekday']/31)

train['xhour']= np.sin(2*3.14*train['thour'] /24)
train['yhour']= np.cos(2*3.14*train['thour'] /24)

train['xminute']= np.sin(2*3.14*train['tminute'] /60)
train['yminute']= np.cos(2*3.14*train['tminute'] /60)

#New features which will boost the performance

#how many ads are launched at a time on under particular site, category, offer etc.
#number of unique merchants on a particular site
#feature number of unique offers on a particular site
#feature on how many sites at a particular time that offer is available




#---------------------------------------------------------Prepare test data-----------------------------------------------------------------------------------------------


test['browserid'] = [re.sub(pattern='^Mozilla$',repl='Mozilla Firefox',string=x) for x in test['browserid']]
test['browserid'] = [re.sub(pattern='^Firefox$',repl='Mozilla Firefox',string=x) for x in test['browserid']]
test['browserid'] = [re.sub(pattern='InternetExplorer',repl='IE',string=x) for x in test['browserid']]
test['browserid'] = [re.sub(pattern='Internet Explorer',repl='IE',string=x) for x in test['browserid']]
test['browserid'] = [re.sub(pattern='Google Chrome',repl='Chrome',string=x) for x in test['browserid']]

test['datetime'] = pd.to_datetime(test['datetime'])

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute

test['xweekday']= np.sin(2*3.14*test['tweekday']/31)
test['yweekday']= np.cos(2*3.14*test['tweekday']/31)

test['xhour']= np.sin(2*3.14*test['thour'] /24)
test['yhour']= np.cos(2*3.14*test['thour'] /24)

test['xminute']= np.sin(2*3.14*test['tminute'] /60)
test['yminute']= np.cos(2*3.14*test['tminute'] /60)


#----------------------------------------------------------------Model Fitting -------------------------------------------------------------------------------

cols = ['siteid','offerid','category','merchant']

for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')
cols_to_use = list(set(train.columns) - set(['ID','datetime','click','tweekday','thour','tminute']))

# catboost accepts categorical variables as indexes
cat_cols = [0,1,2,4,6,7,8]
sampled_train = train.copy()
trainX = sampled_train[cols_to_use]
trainY = sampled_train['click']

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.5)
model = CatBoostClassifier(depth=10, iterations=10, learning_rate=0.1, eval_metric='AUC', random_seed=1)

model.fit(X_train
          ,y_train
          ,cat_features=cat_cols
          ,eval_set = (X_test, y_test)
          ,use_best_model = True
         )


#--------------------------------------------------------------Predict Clicks   ---------------------------------------------------------------------------------

pred = model.predict_proba(test[cols_to_use])[:,1]

sub = pd.DataFrame({'ID':test['ID'],'click':pred})
sub.to_csv('cb_sub1.csv',index=False)




