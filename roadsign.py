

import json
import pandas as pd
import re
from xgboost import XGBClassifier


#from sklearn.metrics import roc_auc_score, make_scorer
#from sklearn.pipeline import Pipeline
#from sklearn.ensemble import VotingClassifier, RandomForestClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.grid_search import GridSearchCV
#from sklearn.feature_selection import SelectFromModel



file_name_1 = "train_data.json"
with open(file_name_1, 'r') as jsonfile1:
    data_dict_1 = json.load(jsonfile1)

file_name_2 = "test_data.json"
with open(file_name_2, 'r') as jsonfile2:
    data_dict_2 = json.load(jsonfile2)

train = pd.DataFrame.from_dict(data_dict_1, orient='index')
train.reset_index(level=0, inplace=True)
train.rename(columns={'index': 'ID'}, inplace=True)


test = pd.DataFrame.from_dict(data_dict_2, orient='index')
test.reset_index(level=0, inplace=True)
test.rename(columns={'index': 'ID'}, inplace=True)


s=','
genres = s.join(train['genres'].values.reshape(1,-1).tolist()[0])

all_genres=[]
for genre in re.findall('[a-z A-Z0-9:]+',genres):
    all_genres.append(genre.split(':')[0])

#List of Unique genres
all_genres=list(set(all_genres))

#pipe_svc = Pipeline([('scl',StandardScaler()),('rf',RandomForestClassifier(n_estimators=460,class_weight='balanced',oob_score=True))])

traindf= train.copy()

#Flattening unstructured data

traindf['total_watch_time'] = 0
for i in range(len(traindf)):
    for title in re.findall('[a-zA-z: 0-9]+', traindf.iloc[i]['genres']):
        traindf.set_value(i, 'total_watch_time', traindf.iloc[i]['total_watch_time'] + int(title.split(':')[1]))



for genre in all_genres[0]:
    traindf[genre+"_watch_time"]=0.0


#creating columns for each genre
#Watch time per genre as percentage of total watch time
for i in range(len(traindf)):
    for genre_watchtime in re.findall('[a-zA-Z: 0-9]+',traindf.iloc[i]['genres']):
        traindf.set_value(i,genre_watchtime.split(':')[0]+"_watch_time",float(genre_watchtime.split(':')[1])/traindf.iloc[i]['total_watch_time'])

#Columns for each day in week
for i in range(1, 8):
    traindf[str(i) + "_watch_time"] = 0

#columns for each hour in a day
for i in range(24):
    traindf[str(i) + "_time_watch_time"] = 0

#expressing watch time as a percentage of total watch time

for i in range(len(traindf)):
    for dow in re.findall('[a-zA-z: 0-9]+', traindf.iloc[i]['dow']):
        traindf.set_value(i, dow.split(':')[0] + "_watch_time", int(dow.split(':')[1])/traindf.iloc[i]['total_watch_time'])

for i in range(len(traindf)):
    for tod in re.findall('[a-zA-z: 0-9]+', traindf.iloc[i]['tod']):
        traindf.set_value(i, tod.split(':')[0] + "_time_watch_time", int(tod.split(':')[1])/traindf.iloc[i]['total_watch_time'])



#Features - number of titles watched,number of genres, number of days in a week, no of times in a day, number of cities from which user logged in

traindf['title_count']=0
traindf['genres_count']=0
traindf['dow_count']=0
traindf['tod_count']=0
traindf['cities_count']=0
def title_count(row):
    return len(re.findall('[\'()a-zA-Z:,& 0-9-]+',row['titles']))

def genres_count(row):
    return len(re.findall('[0-9a-zA-Z :]+',row['genres']))

def dow_count(row):
    return len(re.findall('[0-9a-zA-Z :]+',row['dow']))

def tod_count(row):
    return len(re.findall('[0-9a-zA-Z :]+',row['tod']))

def cities_count(row):
    return len(re.findall('[\'()a-z A-Z0-9:-]+',row['cities']))


traindf['title_count']=traindf.apply(title_count,axis=1)
traindf['genres_count']=traindf.apply(genres_count,axis=1)
traindf['dow_count'] = traindf.apply(dow_count,axis=1)
traindf['tod_count']=traindf.apply(tod_count,axis=1)
traindf['cities_count']= traindf.apply(cities_count,axis=1)



#Tried selecting only top genres contributing to segment 1, reduces comptation time , but no significant effect on score

#traindf['intersted_genres']= traindf['Drama_watch_time'] + traindf['Romance_watch_time'] + traindf['Family_watch_time'] + traindf['Reality_watch_time'] +traindf['TalkShow_watch_time']

#Tried keeping selecting top few cities contributing to segment 1, but again no significant improvement in score

# for i in range(len(traindf)):
#     for city_watchtime in re.findall('[\'()a-zA-Z: 0-9-]+',traindf.iloc[i]['cities']):
#         traindf.set_value(i,city_watchtime.split(':')[0]+"_watch_time",city_watchtime.split(':')[1])



traindf.drop(['titles','genres','ID','dow','tod','cities'],axis=1,inplace=True)
#traindf.to_csv("hostar_flattened.csv",index=False)

X_traindf =  traindf.drop('segment',axis=1)
Y_traindf = traindf['segment'].map({'pos':1,'neg':0})

#-----------------------------------------------------------------------Same features for test data-------------------------------------------------------------------------


#Flattening

test['total_watch_time'] = 0
for i in range(len(test)):
    for title in re.findall('[a-zA-z: 0-9]+', test.iloc[i]['genres']):
        test.set_value(i, 'total_watch_time', test.iloc[i]['total_watch_time'] + int(title.split(':')[1]))

for genre in all_genres[0]:
    test[genre+"_watch_time"]=0.0


for i in range(len(test)):
    for genre_watchtime in re.findall('[a-zA-Z: 0-9]+',test.iloc[i]['genres']):
        test.set_value(i,genre_watchtime.split(':')[0]+"_watch_time",float(genre_watchtime.split(':')[1])/test.iloc[i]['total_watch_time'])

#Watch time per day in week
for i in range(1,8):
    test[str(i)+"_watch_time"]=0

#watch time per hour

for i in range(24):
    test[str(i)+"_time_watch_time"]=0
for i in range(len(test)):
    for tod in re.findall('[a-zA-z: 0-9]+', test.iloc[i]['tod']):
        test.set_value(i, tod.split(':')[0]+"_time_watch_time", int(tod.split(':')[1]))

for i in range(len(test)):
    for dow in re.findall('[a-zA-z: 0-9]+', test.iloc[i]['dow']):
        test.set_value(i, dow.split(':')[0]+"_watch_time", int(dow.split(':')[1]))


test['title_count']=0
test['genres_count']=0
test['dow_count']=0
test['tod_count']=0
test['cities_count']=0

test['title_count']=test.apply(title_count,axis=1)
test['genres_count']=test.apply(genres_count,axis=1)
test['dow_count'] = test.apply(dow_count,axis=1)
test['tod_count']=test.apply(tod_count,axis=1)
test['cities_count']= test.apply(cities_count,axis=1)


#test['intersted_genres'] = test['Drama_watch_time'] + test['Romance_watch_time'] + test['Family_watch_time'] + test['Reality_watch_time'] + test['TalkShow_watch_time']
# for i in range(len(test)):
#     for city_watchtime in re.findall('[\'()a-zA-Z: 0-9-]+',test.iloc[i]['cities']):
#         if city_watchtime.split(':')[0] in test.columns:
#             test.set_value(i,city_watchtime.split(':')[0]+"_watch_time",city_watchtime.split(':')[1])

testdf= test.copy()
test.drop(['titles','genres','ID','dow','tod','cities'],axis=1,inplace=True)


#---------------------------------------------------------Model fitting and prediction-----------------------------------------------------------------






#Tried drop features using feature selection techniques

#sfm = SelectFromModel(model2,threshold = 0.013)
#X_traindf2 = pd.DataFrame(sfm.fit_transform(X_traindf,Y_traindf))
#test2 = pd.DataFrame(sfm.transform(test))

#Tried majority voting on random forest and xgboost

#rf = RandomForestClassifier(n_estimators=460,max_depth=12, max_features=8,class_weight='balanced')
#xgb = XGBClassifier(max_depth=5, n_estimators=460, learning_rate=0.05,scale_pos_weight = 1,min_child_weight = 2,gamma = 0.0,subsample =0.5, colsample_bytree = 0.5,max_delta_step=1,objective = 'binary:logistic')
#model = VotingClassifier(estimators=[('rf',rf),('xgb',xgb)],voting='soft')
#model.fit(X_traindf,Y_traindf)# print("best_params: "+str(rf.best_params_))

#Tried Dimensionality reduction, didnt improve score by much , though reduced computation time

#pca = PCA(n_components=45)
#pca.fit(X_traindf)
#pca.transform(X_traindf)
#pca.transform(test)
#sc.transform(test)

# clf_scorer = make_scorer(roc_auc_score)
# # rfc = RandomForestClassifier(n_estimators=463,oob_score=True)
# param_grid =[ {
#     'rf__max_depth':[3,4,8,12,16],
#     'rf__max_features':['sqrt',10,12,15,20,24]
#
# }]

#Grid Search on Random Forest

# cv_rfc = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, cv=5, scoring=clf_scorer)
#print("best_score: " + str(cv_rfc.best_score_))
#print("best_params: "+ str(cv_rfc.best_params_))


#Parameter tuned using Grid Search -- Best performing model

model = XGBClassifier(max_depth=5, n_estimators=460, learning_rate=0.05,scale_pos_weight = 1,min_child_weight = 2,gamma = 0.0,subsample =0.5, colsample_bytree = 0.5,max_delta_step=1)
model.fit(X_traindf,Y_traindf)




#----------------------------------------------------------------------Predict segment---------------------------------------------------------------------------------------




#Predict probabilities
probabilities = model.predict_proba(test)

#Convert to a dataframe
probabilities = pd.DataFrame(probabilities,columns=['neg','segment'])

#Drop neg class as we are interested in positive segment only

probabilities = probabilities.drop('neg',axis=1)

#concatenate IDs

answer = pd.concat([pd.DataFrame(testdf['ID']),probabilities],axis=1)

#Save results to a file
answer.to_csv('submission.csv',index=False)
