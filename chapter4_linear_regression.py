# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:39:56 2021

@author: Will
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sympy import symbols, diff


teams = pd.read_csv('teams.csv')
teams.tail()
#focus on seasons since 2001
myteams = teams[(teams["yearID"]>2000)]
#subset for teamID, yearID, lgID, G, W, L, R, RA
myteams = myteams[['teamID','yearID','lgID','G','W','L','R','RA']]
myteams.tail()
#create run diff column and win pct column
run_diff = (myteams['R'] - myteams['RA'])
win_pct = myteams['W']/(myteams['W'] + myteams['L'])
myteams.insert(loc=8, column='RD', value=run_diff)
myteams.insert(loc=9, column='Wpct', value = win_pct)
myteams.tail()
plot = sns.regplot(myteams['RD'], myteams['Wpct'], lowess = True, scatter_kws={"color": "black"}, line_kws={"color": "red"})

#lin reg to explore relationship between RD and Wpct
reg = linear_model.LinearRegression()
#reshape values because only one feature
RD = myteams['RD'].values.reshape(-1,1)
linfit = reg.fit(RD, myteams['Wpct'])
print('linear model coeff: {}'
     .format(linfit.coef_))
print('linear model intercept: {:.3f}'
     .format(linfit.intercept_))
print('R-squared score: {:.3f}'
     .format(linfit.score(RD, myteams['Wpct'])))
#predicted win pct for the teams based on the model
linWpct = linfit.predict(RD)
#residuals
linResiduals = myteams['Wpct'] - linWpct
plt.scatter(myteams['RD'], linResiduals)
plt.axhline(y=0, color = 'r', linestyle = '-')
plt.xlabel('run differential')
plt.ylabel('residuals')
#calculate RMSE - average magnitude of the errors
linResiduals.mean()
linRMSE = np.sqrt((linResiduals ** 2).mean())
#percentage of residuals that fall between -RMSE and +RMSE
len(linResiduals[(np.abs(linResiduals) < linRMSE)])/len(linResiduals)
#percentage of residuals that fall between -2*RMSE and 2*RMSE
len(linResiduals[(np.abs(linResiduals) < 2 *linRMSE)])/len(linResiduals)

#pythagorean formula for winning percentage
pytWct = (myteams['R']**2)/(myteams['R']**2 + myteams['RA']**2)
myteams.insert(loc=10, column='pytWpct', value=pytWct)
#pythag formula residuals
pytResiduals = myteams['Wpct'] - myteams['pytWpct']
myteams.insert(loc=11, column='pytResiduals', value=pytResiduals)
#RMSE for pythag formula winning pct
np.sqrt((pytResiduals**2).mean())

#the exponent in the pythag formula
logWratio = np.log(myteams['W']/myteams['L'])
logRatio = np.log(myteams['R']/myteams['RA'])
reshape_logRatio = logRatio.values.reshape(-1,1)
reg = linear_model.LinearRegression(fit_intercept=False)
pytFit = reg.fit(reshape_logRatio, logWratio)
print('linear model coeff (k): {}'
     .format(pytFit.coef_))
#why did boston red sox do poorly compared to their pythag prediction in 2011
gl2011 = pd.read_csv('gl2011.txt', sep=",", header=None)
gl2011.head()
glheaders = pd.read_csv('game_log_header.csv')
gl2011.columns = glheaders.columns
BOS2011 = gl2011.loc[(gl2011["HomeTeam"]=='BOS') | (gl2011["VisitingTeam"]=='BOS' )]
BOS2011 = BOS2011[['VisitingTeam','HomeTeam','VisitorRunsScored','HomeRunsScore']]
BOS2011.head()
#calculate run diff and W/L for each game
def home_team(HomeTeam,HomeRunsScore,VisitorRunsScored):
    if HomeTeam == 'BOS':
        diff = HomeRunsScore - VisitorRunsScored
        return diff
    else:
        diff = VisitorRunsScored - HomeRunsScore
        return diff

BOS2011['ScoreDiff'] = BOS2011.apply(lambda row: home_team(row['HomeTeam'], row['HomeRunsScore'], row['VisitorRunsScored']), axis =1) 
BOS2011['W'] = BOS2011['ScoreDiff'] >0
BOS2011.columns
#view number of wins and losses (True = Win, False = Loss)
BOS2011.W.value_counts()
ax = sns.countplot(x = 'W', data = BOS2011, linewidth = 5, edgecolor = sns.color_palette('dark',3))
#average run diff in W's vs L's 
BOS2011[(BOS2011["W"]==True)].ScoreDiff.mean()
np.abs(BOS2011[(BOS2011["W"]==False)].ScoreDiff.mean())

#create dataframe of all 1 run games
results = gl2011[['VisitingTeam','HomeTeam','VisitorRunsScored','HomeRunsScore']]
results.shape
def winner(HomeRunsScore,VisitorRunsScored,HomeTeam,VisitingTeam):
    if HomeRunsScore > VisitorRunsScored:
        return HomeTeam
    else:
        return VisitingTeam
results['winner'] = results.apply(lambda row: winner(row['HomeRunsScore'], row['VisitorRunsScored'], row['HomeTeam'], row['VisitingTeam']), axis =1)
results.head()
results['diff'] = np.abs(results['VisitorRunsScored'] - results['HomeRunsScore'])
one_run_games = results[(results["diff"]==1)]
#create dataframe of total one run wins for each team
one_run_wins = pd.DataFrame(one_run_games.winner.value_counts())
one_run_wins.head()
one_run_wins = one_run_wins.reset_index()
one_run_wins = one_run_wins.rename(columns={'index':'teamID','winner': 'onerunW'})

#explore relation between Pythag residuals and number of one-run victories
teams2011 = myteams[(myteams['yearID']==2011)]
#change teamID for LA Angels from myteams dataframe to match one_run_wins dataframe
teams2011.index[teams2011['teamID'] == 'LAA']
teams2011.at[2668,'teamID'] = 'ANA'
#check teamID column
teams2011['teamID'].unique
#merge teams dataframe with dataframe that has the number of one run wins per team
teams_2011_final = pd.merge(teams2011, one_run_wins, on='teamID')
plt.scatter(teams_2011_final['onerunW'], teams_2011_final['pytResiduals'])
plt.xlabel('one run wins')
plt.ylabel('Pythagorean residuals')

#why might some teams win more one run games, and thus overperform their pythagorean expected winning percentage
pitching = pd.read_csv('pitching.csv')
top_closers = pitching[(pitching['GF'] > 50) & (pitching['ERA'] <2.5)]
top_closers = top_closers[['playerID','yearID','teamID']]
teams_top_closers = pd.merge(myteams, top_closers)
teams_top_closers['pytResiduals'].mean()
#mean is above 0, means teams with good closers, on average, outperform their pythag win percentage

#how many Runs make up a Win
G, R, A = symbols('G R A', real=True)
f = (G * R**2)/(R**2 + A**2)
diff(f, R)

#function to calculate incremental runs that result in a win
#RS and RA are runs scored/against per game
def IR(RS, RA):
    x = round((RS**2 + RA **2)**2/(2 *RS*RA**2),1)
    return x
IR(5,5)

#RS = np.arange(3,7.5,0.5).tolist()
#RA = np.arange(3,7.5,0.5).tolist()
x = np.array([(x,y) for x in np.arange(3,7.5,0.5) for y in np.arange(3,7.5,0.5)])
df = pd.DataFrame(np.vstack(x))
df.columns =['RS','RA'] 
df['incremental_runs'] = df.apply(lambda row: IR(row['RS'],row['RA']), axis = 1)
table = pd.pivot_table(df, values='incremental_runs',index = ['RS'], columns = ['RA'])


