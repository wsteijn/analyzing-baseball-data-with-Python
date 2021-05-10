# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 19:08:25 2020

@author: Will
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


hof = pd.read_csv("hofbatting2.csv")
#create a variable for the middle of each players career
midCareer = (hof['From'] + hof['To'])/2
#insert this vector into the data frame
hof.insert(loc=25, column='midCareer', value=midCareer)
hof.head()

#create categorical labels out of the continuous mid-career variable
#set the years that divide the categories
breaks=[1800, 1900, 1919, 1941, 1960, 1976, 1993, 2050]
#give labels to each category
labels=["19th Century", "Dead Ball", "Lively Ball","Integration", "Expansion", "Free Agency","Long Ball"]
#use pandas.cut to create the categorical variable
x = pd.cut(hof.midCareer, bins = breaks, labels=labels)
#insert the categorical variable vector into the data frame at the end
hof.insert(26, 'Era', x)

#barplot of the number of hall of famers in each era
hof['Era'].value_counts().plot(kind='bar')
plt.xlabel("Era")
plt.ylabel("Number of hall of famers")

#subset of hall of famers who have over 500 homeruns
hof_500 = hof[(hof["HR"]>=500)]
hof_500 = hof_500.sort_values("OPS", ascending = True)
hof_500.head()
#scatterplot of career OPS values for Hall of Famers with at least 500 home runs
plt.scatter(hof_500['OPS'], hof_500['Name'])

#stripchart of mid career variable to see distribution
sns.stripplot(hof['midCareer'], jitter=True)
plt.xlabel("Mid Career")

#histogram of mid career variable
plt.hist(hof['midCareer'])
plt.xlabel("Mid Career")
#or
sns.distplot(hof['midCareer'], bins = 7, kde=False)

#scatterplot of midCareer and OPS
fig, ax = plt.subplots()
plot = sns.regplot(hof['midCareer'], hof['OPS'], lowess = True, scatter_kws={"color": "black"}, line_kws={"color": "red"})

#scatterplot of obp and slg
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

hof_OPS_1000 = hof[(hof["OPS"]>=1.0)]
 
fig, ax = plt.subplots()
x = np.linspace(0.25, 0.50, 50)
ax = sns.scatterplot(x ='OBP', y= 'SLG', data=hof, color = 'black')
plt.plot(x, .7-x, label = 'OPS = .7')
plt.plot(x, 0.8-x, label = 'OPS = .8')
plt.plot(x, 0.9-x, label = 'OPS = .9')
plt.plot(x, 1.0 - x, label = 'OPS = 1.0')
plt.xlim([0.25, 0.50])
plt.ylim([0.28, 0.75])
plt.xlabel('On-Base Percentage')
plt.ylabel('Slugging Percentage')
plt.legend(loc = 'upper left')
label_point(hof_OPS_1000.OBP, hof_OPS_1000.SLG, hof_OPS_1000.Name, plt.gca()) 
plt.show()    

#Comparing Ruth, Aaron, Bonds, and A-Rod
people = pd.read_csv('People.csv')

#function for getting the id and birth year 
def getInfo(firstName, lastName):
    playerLine = people[(people['nameFirst']==firstName) & (people['nameLast']==lastName)]
    name_code = playerLine.iloc[0]['playerID']
    birthYear = playerLine.iloc[0]['birthYear']
    birthMonth = playerLine.iloc[0]['birthMonth']
    byear=0
    if birthMonth <= 6:
        byear = birthYear + 1
    else:
        byear = birthYear
    return name_code, byear
aaron_info = getInfo('Hank', 'Aaron')
ruth_info = getInfo("Babe", "Ruth")
bonds_info = getInfo("Barry", "Bonds")
arod_info = getInfo("Alex", "Rodriguez")
aaron_info

#creating the player data frames
batting = pd.read_csv("Batting.csv")

aaron_data =  batting[(batting['playerID']==aaron_info[0])]
aaron_age = aaron_data['yearID'] - aaron_info[1]
aaron_data.insert(loc=22, column='Age', value=aaron_age)

ruth_data =  batting[(batting['playerID']==ruth_info[0])]
ruth_age = ruth_data['yearID'] - ruth_info[1]
ruth_data.insert(loc=22, column='Age', value=ruth_age)

bonds_data =  batting[(batting['playerID']==bonds_info[0])]
bonds_age = bonds_data['yearID'] - bonds_info[1]
bonds_data.insert(loc=22, column='Age', value=bonds_age)

arod_data =  batting[(batting['playerID']==arod_info[0])]
arod_age = arod_data['yearID'] - arod_info[1]
arod_data.insert(loc=22, column='Age', value=arod_age)

#make the plots
fig, ax = plt.subplots()
plt.plot(aaron_data['Age'],aaron_data['HR'].cumsum(), label = 'Hank Aaron')
plt.plot(ruth_data['Age'],ruth_data['HR'].cumsum(), label = 'Babe Ruth')
plt.plot(bonds_data['Age'],bonds_data['HR'].cumsum(), label = 'Barry Bonds')
plt.plot(arod_data['Age'],arod_data['HR'].cumsum(), label = 'Alex Rodriguez')
plt.xlabel('Age')
plt.ylabel('Home runs')
plt.legend(loc = 'upper left')
plt.show()   


#the 1998 home run race
data1998 = pd.read_csv('all1998.csv')
fields = pd.read_csv('fields.csv')
data1998.columns = fields['Header']
retro_ids = pd.read_csv('retrosheetIDs.csv')
retro_ids.columns = retro_ids.columns.str.strip()
#extract batting data for McGwire and Sosa
sosa = retro_ids.loc[(retro_ids['FIRST'] == 'Sammy') & (retro_ids['LAST'] == 'Sosa')]
sosa_id = sosa.iloc[0]['ID']
mcgwire = retro_ids.loc[(retro_ids['FIRST'] == 'Mark') & (retro_ids['LAST'] == 'McGwire')]
mcgwire_id = mcgwire.iloc[0]['ID']
sosa_data = data1998.loc[(data1998['BAT_ID'] == sosa_id)]
mac_data = data1998.loc[(data1998['BAT_ID'] == mcgwire_id)]
#extracting the variables
#GAME_ID identifies the game location and date - ARI199805110 = arizona, may 11, 1998
#EVENT_CD contains the outcome of the batting play. 23 = homerun
#output is a new data frame containing the date and the cumulative number of homeruns for all plate appearances
def is_homerun(x):
    if x==23:
        return 1
    else:
        0
        
def createData(df):
    df['Date'] = pd.to_datetime(df.GAME_ID.str.slice(3,11), format="%Y-%m-%d")
    df = df.sort_values('Date', ascending = True)
    df['HR'] = df['EVENT_CD'].apply(lambda x: is_homerun(x)) 
    df['HR'] = df['HR'].fillna(0)
    df['cumHR'] = df['HR'].cumsum()
    return df[['Date', 'cumHR']]
sosa_hr = createData(sosa_data)
mac_hr = createData(mac_data)
sosa_hr.info()
hr_record = pd.DataFrame({'Date':mac_hr['Date'], 'HR':[62]*len(mac_data)})
#make the plot
fig, ax = plt.subplots()
plt.plot(sosa_hr['Date'], sosa_hr['cumHR'], label = 'Sammy Sosa')
plt.plot(mac_hr['Date'], mac_hr['cumHR'], label = 'Mark McGwire')
plt.plot(hr_record['Date'], hr_record['HR'], label = 'Single Season Record')
plt.legend()
plt.show() 

  
