from sklearn.ensemble import RandomForestRegressor as rf
import pandas as pd
from pandas import DataFrame
from scipy.stats.stats import pearsonr as pearson
from sklearn import cross_validation, preprocessing
import numpy as np
import matplotlib.pyplot as plt
import string
import re 

df = pd.read_csv('./test.csv')

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    return np.nan

#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

df['Title']=df.apply(replace_titles, axis=1)

df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

f = {}
le = preprocessing.LabelEncoder()

for s in xrange(0,520,10):
	r = {}
	x = df[df.Fare>=s]
	x = x[x.Fare<=s+10]
	for col in df.columns:
		r[col] = x[col].max()
	f[int(s/10)] = r

for i in xrange(len(df)):
	pclass = df.ix[i]['Fare']
	if pd.isnull(pclass):
		clas = df.ix[i]['Pclass']
		sub = df[df.Pclass==clas]
		pclass = sub.Fare.max()
	for col in df.columns:
		if pd.isnull(df[col][i]):
			val = f[int(pclass/10)][col]
			df[col][i] = val

df['Fare_Per_Person'] = df["Fare"] / (df["SibSp"] + df["Parch"] + 1)
# Create a feature for the deck
df['Deck'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())
df['Deck'] = pd.factorize(df['Deck'])[0]
 
# Create binary features for each deck
decks = pd.get_dummies(df['Deck']).rename(columns=lambda x: 'Deck_' + str(x))
df = pd.concat([df, decks], axis=1)
 
# Create feature for the room number
df['Room'] = df['Cabin'].map( lambda x : re.compile("([0-9]+)").search(x).group()).astype(int) + 1

# print df['Cabin']

for col in df.select_dtypes(include=['O']):
	le.fit(df[col])
	print le.classes_
	df[col] = le.transform(df[col])
	df[col] = df[col].astype(int)

df.to_csv('new_test.csv')