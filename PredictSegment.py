import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, make_scorer
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

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
print("read files")
s=','
genres = s.join(train['genres'].values.reshape(1,-1).tolist()[0])
import re
all_genres=[]
for genre in re.findall('[a-z A-Z0-9:]+',genres):
    all_genres.append(genre.split(':')[0])
all_genres=list(set(all_genres))

all_genres = pd.DataFrame(all_genres)
all_genres[0].unique().shape
s=','
# cities = s.join(train['cities'].values.reshape(1,-1).tolist()[0])
# import re
# all_cities=[]
# for genre in re.findall('[\'()a-z A-Z0-9:-]+',cities):
#     all_cities.append(genre.split(':')[0])
#
# all_cities=list(set(all_cities))
#
# all_cities = pd.DataFrame(all_cities)

pipe_svc = Pipeline([('scl',StandardScaler()),('rf',RandomForestClassifier(n_estimators=460,class_weight='balanced',oob_score=True))])

traindf= train.copy()
# for genre in all_genres[0]:
#     traindf[genre]=0
for genre in all_genres[0]:
    traindf[genre+"_watch_time"]=0
# for city in all_cities[0]:
#     traindf[city+"_watch_time"]=0

# for i in range(len(traindf)):
#     for genre in re.findall('[a-zA-z: 0-9]+',traindf.iloc[i]['genres']):
#         traindf.set_value(i,genre.split(':')[0],traindf.iloc[i][genre.split(':')[0]]+1)
for i in range(len(traindf)):
    for genre_watchtime in re.findall('[a-zA-Z: 0-9]+',traindf.iloc[i]['genres']):
        traindf.set_value(i,genre_watchtime.split(':')[0]+"_watch_time",genre_watchtime.split(':')[1])

#traindf['intersted_genres']= traindf['Drama_watch_time'] + traindf['Romance_watch_time'] + traindf['Family_watch_time'] + traindf['Reality_watch_time'] +traindf['TalkShow_watch_time']

# for i in range(len(traindf)):
#     for city_watchtime in re.findall('[\'()a-zA-Z: 0-9-]+',traindf.iloc[i]['cities']):
#         print(city_watchtime + " -> ")
#         print(city_watchtime.split(':')[0] + "," + city_watchtime.split(':')[1])
#         traindf.set_value(i,city_watchtime.split(':')[0]+"_watch_time",city_watchtime.split(':')[1])

traindf['title_count']=0
traindf['genres_count']=0
traindf['dow_count']=0
traindf['tod_count']=0
def title_count(row):
    return len(re.findall('[\'()a-zA-Z:,& 0-9-]+',row['titles']))

def genres_count(row):
    return len(re.findall('[0-9a-zA-Z :]+',row['genres']))

def dow_count(row):
    return len(re.findall('[0-9a-zA-Z :]+',row['dow']))

def tod_count(row):
    return len(re.findall('[0-9a-zA-Z :]+',row['tod']))
print("train is ready")
traindf['title_count']=traindf.apply(title_count,axis=1)
traindf['genres_count']=traindf.apply(genres_count,axis=1)
traindf['dow_count'] = traindf.apply(dow_count,axis=1)
traindf['tod_count']=traindf.apply(tod_count,axis=1)
# traindf['early'] = 0
# traindf['morning'] = 0
# traindf['afternoon'] = 0
# traindf['evening'] = 0
# traindf['night'] = 0

for i in range(1,8):
    traindf[str(i)+"_watch_time"]=0
for i in range(24):
    traindf[str(i)+"_time_watch_time"]=0
for i in range(len(traindf)):
    for dow in re.findall('[a-zA-z: 0-9]+', traindf.iloc[i]['dow']):
        traindf.set_value(i, dow.split(':')[0]+"_watch_time", int(dow.split(':')[1]))

for i in range(len(traindf)):
    for tod in re.findall('[a-zA-z: 0-9]+', traindf.iloc[i]['tod']):
        traindf.set_value(i, tod.split(':')[0]+"_time_watch_time", int(tod.split(':')[1]))



# for i in range(len(traindf)):
#     for tod in re.findall('[a-zA-z: 0-9]+', traindf.iloc[i]['tod']):
#         if int(tod.split(':')[0]) <= 5 or int(tod.split(':')[0]) >= 23:
#             traindf.set_value(i, 'early', int(tod.split(':')[1])+int(traindf.iloc[i]['early']))
#         elif int(tod.split(':')[0]) <= 10:
#             traindf.set_value(i, 'morning', int(tod.split(':')[1])+int(traindf.iloc[i]['morning']))
#         elif int(tod.split(':')[0]) <= 14:
#             traindf.set_value(i, 'afternoon', int(tod.split(':')[1])+int(traindf.iloc[i]['afternoon']))
#         elif int(tod.split(':')[0]) <= 18:
#             traindf.set_value(i, 'evening', int(tod.split(':')[1])+int(traindf.iloc[i]['evening']))
#         else:
#              traindf.set_value(i, 'night', int(tod.split(':')[1])+int(traindf.iloc[i]['night']))

traindf['total_watch_time'] = 0
for i in range(len(traindf)):
    for title in re.findall('[a-zA-z: 0-9]+', traindf.iloc[i]['genres']):
        traindf.set_value(i, 'total_watch_time', traindf.iloc[i]['total_watch_time'] + int(title.split(':')[1]))

traindf.drop(['titles','genres','ID','dow','tod','cities'],axis=1,inplace=True)
# traindf.to_csv("hostar_flattened.csv",index=False)
X_traindf =  traindf.drop('segment',axis=1)
Y_traindf = traindf['segment'].map({'pos':1,'neg':0})

##count dow and tod also
#svc = SVC(kernel='rbf',probability=True,C=1.0)
# sc = StandardScaler()
# sc.fit(X_traindf)
# sc.transform(X_traindf)
print("donr scaling")
# clf_scorer = make_scorer(roc_auc_score)
# # rfc = RandomForestClassifier(n_estimators=463,oob_score=True)
# param_grid =[ {
#     'rf__max_depth':[3,4,8,12,16],
#     'rf__max_features':['sqrt',10,12,15,20,24]
#
# }]

# cv_rfc = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, cv=5, scoring=clf_scorer)
# rf_features = XGBClassifier(max_depth=3, n_estimators=500, learning_rate=0.05,objective='binary:logistic')
# #rf_features= RandomForestClassifier(n_estimators=500,max_depth=12, max_features=8,class_weight='balanced')
# #rf_features.fit_
# #rf= LogisticRegression(C=100.0)
# rf_features.fit(X_traindf,Y_traindf)
#pca = PCA(n_components=45)
# pca.fit(X_traindf)
#pca.transform(X_traindf)
print("done_fitting")

# for genre in all_genres[0]:
#     test[genre]=0
for genre in all_genres[0]:
    test[genre+"_watch_time"]=0
# for city in all_cities[0]:
#     test[city+"_watch_time"]=0

# for i in range(len(test)):
#     for genre in re.findall('[a-zA-z: 0-9]+',test.iloc[i]['genres']):
#         test.set_value(i,genre.split(':')[0],test.iloc[i][genre.split(':')[0]]+1)
for i in range(len(test)):
    for genre_watchtime in re.findall('[a-zA-Z: 0-9]+',test.iloc[i]['genres']):
        test.set_value(i,genre_watchtime.split(':')[0]+"_watch_time",genre_watchtime.split(':')[1])

#test['intersted_genres'] = test['Drama_watch_time'] + test['Romance_watch_time'] + test['Family_watch_time'] + test['Reality_watch_time'] + test['TalkShow_watch_time']
# for i in range(len(test)):
#     for city_watchtime in re.findall('[\'()a-zA-Z: 0-9-]+',test.iloc[i]['cities']):
#         if city_watchtime.split(':')[0] in test.columns:
#             print(city_watchtime + " -> ")
#             test.set_value(i,city_watchtime.split(':')[0]+"_watch_time",city_watchtime.split(':')[1])
#             print(city_watchtime.split(':')[0]+","+city_watchtime.split(':')[1])

#
#
#

test['title_count']=0
test['genres_count']=0
test['dow_count']=0
test['tod_count']=0

#
# test['early'] = 0
# test['morning'] = 0
# test['afternoon'] = 0
# test['evening'] = 0
# test['night'] = 0
# for i in range(len(test)):
#     for tod in re.findall('[a-zA-z: 0-9]+', test.iloc[i]['tod']):
#         if int(tod.split(':')[0]) <= 5 or int(tod.split(':')[0]) >= 23:
#             test.set_value(i, 'early', int(tod.split(':')[1])+int(test.iloc[i]['early']))
#         elif int(tod.split(':')[0]) <= 10:
#             test.set_value(i, 'morning', int(tod.split(':')[1])+int(test.iloc[i]['morning']))
#         elif int(tod.split(':')[0]) <= 14:
#             test.set_value(i, 'afternoon', int(tod.split(':')[1])+int(test.iloc[i]['afternoon']))
#         elif int(tod.split(':')[0]) <= 18:
#             test.set_value(i, 'evening', int(tod.split(':')[1])+int(test.iloc[i]['evening']))
#         else:
#             test.set_value(i, 'night', int(tod.split(':')[1])+int(test.iloc[i]['night']))
for i in range(1,8):
    test[str(i)+"_watch_time"]=0

for i in range(24):
    test[str(i)+"_time_watch_time"]=0
for i in range(len(test)):
    for tod in re.findall('[a-zA-z: 0-9]+', test.iloc[i]['tod']):
        test.set_value(i, tod.split(':')[0]+"_time_watch_time", int(tod.split(':')[1]))

for i in range(len(test)):
    for dow in re.findall('[a-zA-z: 0-9]+', test.iloc[i]['dow']):
        test.set_value(i, dow.split(':')[0]+"_watch_time", int(dow.split(':')[1]))
test['total_watch_time'] = 0
for i in range(len(test)):
    for title in re.findall('[a-zA-z: 0-9]+', test.iloc[i]['genres']):
        test.set_value(i, 'total_watch_time', test.iloc[i]['total_watch_time'] + int(title.split(':')[1]))

test['title_count']=test.apply(title_count,axis=1)
test['genres_count']=test.apply(genres_count,axis=1)
test['dow_count'] = test.apply(dow_count,axis=1)
test['tod_count']=test.apply(tod_count,axis=1)
testdf= test.copy()
test.drop(['titles','genres','ID','dow','tod','cities'],axis=1,inplace=True)
print("now predicting")
# pca.transform(test)
# sc.transform(test)
#rf = RandomForestClassifier(n_estimators=460,max_depth=12, max_features=8,class_weight='balanced')
#xgb
model2 = XGBClassifier(max_depth=5, n_estimators=460, learning_rate=0.05,scale_pos_weight = 1,min_child_weight = 2,gamma = 0.0,subsample =0.5, colsample_bytree = 0.5,max_delta_step=1)
#model = VotingClassifier(estimators=[('rf',rf),('xgb',xgb)],voting='soft')
sfm = SelectFromModel(model2,threshold = 0.013)
X_traindf2 = pd.DataFrame(sfm.fit_transform(X_traindf,Y_traindf))
test2 = pd.DataFrame(sfm.transform(test))
print("now grid serach")
model = XGBClassifier(max_depth=5, n_estimators=460, learning_rate=0.05,scale_pos_weight = 1,min_child_weight = 2,gamma = 0.0,subsample =0.5, colsample_bytree = 0.5,max_delta_step=1)

model.fit(X_traindf2,Y_traindf)# print("best_params: "+str(rf.best_params_))
probabilities = model.predict_proba(test2)
print probabilities
probabilities1 = pd.DataFrame(probabilities,columns=['neg','segment'])
probabilities1 = probabilities1.drop('neg',axis=1)
answer = pd.concat([pd.DataFrame(testdf['ID']),probabilities1],axis=1)
answer.to_csv('segmentspredanswer78.csv',index=False)
# print("best_score: " + str(cv_rfc.best_score_))
# print("best_params: "+str(cv_rfc.best_params_))
#
