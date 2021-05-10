# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:50:09 2021

@author: Will
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import math

#RUNS EXPECTANCY MATRIX

#Runs scored in the remainder of the inning
data2011 = pd.read_csv('all2011.csv', header = None)
fields = pd.read_csv('fields.csv')
data2011.columns = fields['Header']
#runs column - number of runs scored in the game at the beginning of each plate appearance
data2011['RUNS'] = data2011['AWAY_SCORE_CT'] + data2011['HOME_SCORE_CT']
#unique ID for each half inning of every game during the season
data2011['HALF_INNING']=data2011.apply(lambda x:'%s_%s_%s' % (x['GAME_ID'],x['INN_CT'],x['BAT_HOME_ID']),axis=1)
#runs scored variable created that givesthe number of runs scored for each play
def runs_scored(BAT,RUN1,RUN2,RUN3):
    runs = 0
    if BAT > 3:
        runs = runs + 1
    if RUN1 >3:
        runs = runs + 1
    if RUN2 > 3:
        runs = runs + 1
    if RUN3 > 3:
        runs = runs + 1
    return runs
data2011['RUNS_SCORED'] = data2011.apply(lambda row: runs_scored(row['BAT_DEST_ID'], row['RUN1_DEST_ID'], row['RUN2_DEST_ID'], row['RUN3_DEST_ID']), axis =1) 
#find how many runs are scored in each half inning
RUNS_SCORED_INNING = data2011.groupby('HALF_INNING')['RUNS_SCORED'].agg('sum')
RUNS_SCORED_INNING = RUNS_SCORED_INNING.to_frame().reset_index()
#find the total game runs at the beginning of each half-inning
#get a dataframe of the indices of the first instance of each half-inning, and the ID of each half-inning
unique_innings = pd.DataFrame(data2011.HALF_INNING.drop_duplicates())
#merge with full data based on the indices of the first instance of each unique half inning
RUNS_SCORED_START = unique_innings.merge(data2011, left_index = True, right_index = True)
#extract the two columns that are needed fromthe dataframe
RUNS_SCORED_START = RUNS_SCORED_START[["HALF_INNING_x", "RUNS"]]
#rename the first column to drop the _x
RUNS_SCORED_START.rename(columns={'HALF_INNING_x': 'HALF_INNING'}, inplace = True)
RUNS_SCORED_START.columns
RUNS_SCORED_INNING.columns
#create new dataframe with the maximum number of runs scored
RUNS_SCORED_FINAL = RUNS_SCORED_START.merge(RUNS_SCORED_INNING,left_on='HALF_INNING', right_on='HALF_INNING')
RUNS_SCORED_FINAL.columns
#x= column with max runs
RUNS_SCORED_FINAL['x'] = RUNS_SCORED_FINAL['RUNS'] + RUNS_SCORED_FINAL['RUNS_SCORED']
#delete the excess columns - only want innings and max runs
RUNS_SCORED_FINAL.drop(['RUNS','RUNS_SCORED'], inplace = True, axis=1)
#merge max data with the complete dataframe
data2011= pd.merge(data2011, RUNS_SCORED_FINAL)
#rename the last column "MAX RUNS"
data2011.rename(columns={'x': 'MAX_RUNS'}, inplace=True)
#now the runs scored in the remainder of hte inning an be computed by taking the difference of max.runs and runs
data2011['RUNS_ROI'] = data2011['MAX_RUNS'] - data2011['RUNS']


#creating the matrix

#create three binary variables to determine if each base is occupied or empty
def isNaN(num):
    return num == num
RUNNER1= data2011.apply(lambda row: isNaN(row['BASE1_RUN_ID']), axis =1) 
RUNNER2= data2011.apply(lambda row: isNaN(row['BASE2_RUN_ID']), axis =1) 
RUNNER3= data2011.apply(lambda row: isNaN(row['BASE3_RUN_ID']), axis =1) 
#convert true/false to 1/0
RUNNER1 = RUNNER1.astype(int)
RUNNER2 = RUNNER2.astype(int)
RUNNER3 = RUNNER3.astype(int)
#add these columns to the dataframe to make it easier to make the state column
data2011.insert(loc=102, column='RUNNER1', value=RUNNER1)
data2011.insert(loc=103, column='RUNNER2', value=RUNNER2)
data2011.insert(loc=104, column='RUNNER3', value=RUNNER3)
#create a state variable commbining the runner indicators and the number of outs
data2011['STATE']=data2011.apply(lambda row:'%s%s%s_%s' % (row['RUNNER1'],row['RUNNER2'],row['RUNNER3'],row['OUTS_CT']),axis=1)
#only want to consider plays where there is a change in runners on base, number of outs, or runs scored
#3 new variables:
#NRUNNER1
#NRUNNER2
#NRunner3
#these variables indicate if 1b/2b/3b are occupied after the play
#variable NOUTS is number of outs after the play
#variable RUNS_SCORED is the number of runs scored on the play
data2011['RUN1_DEST_ID']
NRUNNER1 = data2011.apply(lambda row: ((row['RUN1_DEST_ID'] == 1) | (row['BAT_DEST_ID'] == 1)), axis = 1).astype(int)
NRUNNER2 = data2011.apply(lambda row: ((row['RUN1_DEST_ID'] == 2) | (row['RUN2_DEST_ID'] == 2) | (row['BAT_DEST_ID'] == 2)), axis = 1).astype(int)
NRUNNER3 = data2011.apply(lambda row: ((row['RUN1_DEST_ID'] == 3) | (row['RUN2_DEST_ID'] == 3) | (row['RUN3_DEST_ID'] == 3) | (row['BAT_DEST_ID'] == 3)), axis = 1).astype(int)
NOUTS = data2011['OUTS_CT'] + data2011['EVENT_OUTS_CT']
#add these columns to the dataframe to make it easier to make the state column
data2011.insert(loc=105, column='NRUNNER1', value=NRUNNER1)
data2011.insert(loc=106, column='NRUNNER2', value=NRUNNER2)
data2011.insert(loc=107, column='NRUNNER3', value=NRUNNER3)
data2011.insert(loc=108, column='NOUTS', value=NOUTS)
#create a new_state variable giving the runners on each base and the number of outs after the play
data2011['NEW_STATE']=data2011.apply(lambda row:'%s%s%s_%s' % (row['NRUNNER1'],row['NRUNNER2'],row['NRUNNER3'],row['NOUTS']),axis=1)
#by use of the subset fucntion, find subset where there is a change between State and New State or there are runs scored on the play
data2011 = data2011[(data2011["STATE"] != data2011['NEW_STATE']) | (data2011['RUNS_SCORED'] > 0)]
#compute number of outs for each half inning
data_outs = data2011.groupby(['HALF_INNING'], as_index=False)['EVENT_OUTS_CT'].sum()
data_outs.rename(columns={'EVENT_OUTS_CT': 'OUTS_INNING'}, inplace = True)
#merge with full data
data2011 = data2011.merge(data_outs, left_on='HALF_INNING', right_on='HALF_INNING')
#subset of just the innings that end with 3 outs
data2011C = data2011[(data2011["OUTS_INNING"]==3)]
#the expected number of runs scored in the remainder of the inning (runs expectancy) is computed 
#for each of the 24 bases/outs combinatinos
RUNS = data2011C.groupby(['STATE'], as_index = False)['RUNS_ROI'].mean()
RUNS['OUTS']= RUNS.STATE.str.slice(4)
RUNS['STATE'] =RUNS.STATE.str.slice(0,3)
RUNS = RUNS.sort_values(by = ['OUTS','STATE'])
RUNS.set_index('STATE', inplace=True)
RUNS = RUNS.pivot(columns = 'OUTS')

#Measuring success of a batting play

#RUNS VALUE = RUNS_new_state - RUNS_old_state + RUNS_scored_on_play
RUNS_POTENTIAL = data2011C.groupby(['STATE'], as_index = False)['RUNS_ROI'].mean()
new_states = ['000_3','001_3','010_3','011_3','100_3','101_3','110_3','111_3']
new_runs = [0]*8
#add these new states (situations with 3 outs) and their run expectancy to runs_potential dataframe
df2 = pd.DataFrame(list(zip(new_states,new_runs)), columns = RUNS_POTENTIAL.columns)
RUNS_POTENTIAL = RUNS_POTENTIAL.append(df2, ignore_index=True)
#create a dictionary where the keys are the states and the values are the run expectancy 
runs_dictionary = dict(RUNS_POTENTIAL.values)
#use the dictionary to get the runs value for the state at the beginning of the play and the state at the end of the play
data2011['RUNS_STATE'] = data2011['STATE'].map(runs_dictionary)
data2011['RUNS_NEW_STATE'] = data2011['NEW_STATE'].map(runs_dictionary)
#find the value of the plate appearance
data2011['RUNS_VALUE'] = data2011['RUNS_NEW_STATE'] - data2011['RUNS_STATE'] + data2011['RUNS_SCORED']

#Albert Pujols
#get albert pujols' id
roster = pd.read_csv('roster.csv')
roster.columns
albert_id = roster[(roster['First.Name'] == 'Albert') & (roster['Last.Name'] == 'Pujols')]
albert_id = albert_id.iloc[0]['Player.ID']
#find all the plate appearances for pujols
albert = data2011[(data2011["BAT_ID"]==albert_id)]
albert = albert[(albert['BAT_EVENT_FL'] == 'T')]
#look at his first to plate appearances
albert[0:2][['STATE','NEW_STATE','RUNS_VALUE']]
#dataset of just state, new state, and run value
albert_1 = albert.loc[:,['STATE','NEW_STATE','RUNS_VALUE']]
#view the counts of the states of the runners
albert['RUNNERS']=albert.STATE.str.slice(0,3)
albert['RUNNERS'].value_counts()
#striplot 
sns.stripplot(albert['RUNNERS'], albert['RUNS_VALUE'],jitter=True)
#runs_value for each configuration of runners onbase
A_runs = albert.groupby(['RUNNERS'], as_index = False)['RUNS_VALUE'].sum()
A_pa = albert.groupby(['RUNNERS'], as_index = False)['RUNS_VALUE'].count()
A_pa.rename(columns={'RUNS_VALUE': 'PA'}, inplace = True)

A = A_pa.merge(A_runs, left_on='RUNNERS', right_on='RUNNERS')
pujols_runs = A['RUNS_VALUE'].sum()


#opportunity and success for all hitters
data2011b = data2011[(data2011['BAT_EVENT_FL'] == 'T')]
runs_sums = data2011b.groupby(['BAT_ID'], as_index=False)['RUNS_VALUE'].sum()
runs_sums.rename(columns={'RUNS_VALUE': 'RUNS'}, inplace = True)
runs_pa = data2011b.groupby(['BAT_ID'], as_index = False)['RUNS_VALUE'].count()
runs_pa.rename(columns={'RUNS_VALUE': 'PA'}, inplace = True)
runs_start = data2011b.groupby(['BAT_ID'], as_index = False)['RUNS_STATE'].sum()
runs_start.rename(columns={'RUNS_STATE': 'RUNS_START'}, inplace = True)
runs = runs_sums.merge(runs_pa)
runs = runs.merge(runs_start)
x = runs['RUNS'].max()
runs.loc[runs['RUNS'] == x]
x_1 = runs['RUNS'].min()
runs.loc[runs['RUNS'] == x_1]
runs['runs_percent'] = runs['RUNS']/runs['RUNS_START']
x_2 = runs['runs_percent'].max()
runs.loc[runs['runs_percent']==x_2]

#subset runs for batters with at least 400 plate appearances
runs400 = runs[(runs['PA'] >= 400)]

#scatterplot of run opportunity against runs value
plot = sns.regplot(runs400['RUNS_START'], runs400['RUNS'], lowess = True, scatter_kws={"color": "black"}, line_kws={"color": "red"})
plt.axhline(y=0, color = 'g', linestyle = '-')

#subset runs400 for top players that created over 40 runs for their team
runs400_top = runs400[(runs400['RUNS'] >= 40)]
roster2011 = pd.read_csv('roster.csv')
runs400_top = runs400_top.merge(roster2011,left_on='BAT_ID', right_on='Player.ID' )
runs400_top.rename(columns={'Last.Name': 'last_name'}, inplace = True)
runs400_top.columns
#plot with top batters labeled
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))
fig, ax = plt.subplots()
ax = sns.regplot(runs400['RUNS_START'], runs400['RUNS'], lowess = True, scatter_kws={"color": "black"}, line_kws={"color": "red"})
plt.axhline(y=0, color = 'black', linestyle = '-')
label_point(runs400_top.RUNS_START, runs400_top.RUNS, runs400_top.last_name, plt.gca()) 
plt.show()    


#Postion in the batting lineup

#function to find a player's batting position
def get_batting_position(batter):
    TB = data2011[(data2011['BAT_ID'] == batter)]
    pos =  TB['BAT_LINEUP_ID'].mode()[0]
    return pos
runs400['position']=runs400.apply(lambda row: get_batting_position(row['BAT_ID']),axis=1)
#plot per position in batting order
groups = runs400.groupby('position')
for name, group in groups:
    plt.plot(group['RUNS_START'],group['RUNS'], marker = 'o', linestyle="", label=name)
sns.regplot(runs400['RUNS_START'],runs400['RUNS'], lowess=True)
plt.legend()

#adding Pujols as a distinct point - red point in graph
AP = runs400[(runs400['BAT_ID'] == albert_id)]
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))
fig, ax = plt.subplots()
ax = sns.regplot(runs400['RUNS_START'], runs400['RUNS'], lowess = True, scatter_kws={"color": "black",'s':runs400['position']*2}, line_kws={"color": "red"})
plt.plot(AP['RUNS_START'],AP['RUNS'], 'ro', markersize = 20)
plt.axhline(y=0, color = 'black', linestyle = '-')
label_point(runs400_top.RUNS_START, runs400_top.RUNS, runs400_top.last_name, plt.gca()) 
plt.show()  


#Run Values of Different Base Hits

#value of a homerun
d_homerun = data2011[(data2011['EVENT_CD']==23)]
counts = d_homerun['STATE'].value_counts()
total_ABs = counts.sum()
frequencies = counts/total_ABs
#the fraction of home runs hit with the bases empty is .269 + .178 + .138
#histogram of run values for all homeruns
mean_homerun = d_homerun['RUNS_VALUE'].mean()
plt.hist(d_homerun['RUNS_VALUE'], density=True, bins=len(counts), label="Data")
plt.axvline(x=mean_homerun, color = 'red', label = 'Mean Runs Value')
plt.legend(loc="upper right")
plt.ylabel('Probability density')
plt.xlabel('Homerun Runs Value')
plt.title("Histogram")
#state of most valuable homeruns
max_value_homerun = d_homerun['RUNS_VALUE'].max()
#dataframe of all the instances of max value homerun
max_state_homerun = d_homerun.loc[d_homerun['RUNS_VALUE'] == max_value_homerun]
max_state_homerun = max_state_homerun.loc[:,['STATE','NEW_STATE','RUNS_VALUE']]
max_state_homerun['STATE']
#mean value of a homerun = 1.39
mean_homerun = d_homerun['RUNS_VALUE'].mean()

#value of a single
d_single = data2011[(data2011['EVENT_CD']==20)]
single_state_counts = d_single['STATE'].value_counts()
mean_single = d_single['RUNS_VALUE'].mean()
plt.hist(d_single['RUNS_VALUE'], density=True, bins=2*len(single_state_counts), label="Data")
plt.axvline(x=mean_single, color = 'red', label = 'Mean Runs Value')
plt.legend(loc="upper right")
plt.ylabel('Probability density')
plt.xlabel('Single Runs Value')
plt.title("Histogram");
#most valuable single -single with bases loaded and the leftfielder made an error so the runner got to 3b
max_value_single = d_single['RUNS_VALUE'].max()
max_state_single = d_single.loc[d_single['RUNS_VALUE'] == max_value_single]
max_state_single = max_state_single.loc[:,['STATE','NEW_STATE','RUNS_VALUE']]
max_state_single['STATE']
#least valuable single - runner on 2b who was thrown out at the plate as the result of the single
min_value_single = d_single['RUNS_VALUE'].min()
min_state_single = d_single.loc[d_single['RUNS_VALUE'] == min_value_single]
min_state_single = min_state_single.loc[:,['STATE','NEW_STATE','RUNS_VALUE']]
min_state_single['STATE'][:1]

#value of base stealing (4 = SB, 6 = CS)
stealing = data2011[(data2011['EVENT_CD']==6) | (data2011['EVENT_CD']==4)]
stealing_counts = stealing['EVENT_CD'].value_counts()
stealing_states = stealing['STATE'].value_counts()
plt.hist(stealing['RUNS_VALUE'], density=True, bins=2*len(stealing_states), label="Data")
plt.legend(loc="upper right")
plt.ylabel('Probability density')
plt.xlabel('Single Runs Value')
plt.title("Histogram")
#subset by state - runner on first, one out
stealing_1001 = stealing[(stealing['STATE'] == '100_1')]
stealing_1001 = stealing_1001.loc[:,['STATE','NEW_STATE','RUNS_VALUE']]
#frequency of new state variable
stealing_1001['NEW_STATE'].value_counts()
#mean runs value
stealing_1001['RUNS_VALUE'].mean()


#EXERCISES

#1
#mean run value for single, double, triple, and homerun
for i in range(20,24):
    hit = data2011[(data2011['EVENT_CD']== i)]
    hit_mean = hit['RUNS_VALUE'].mean()
    print (i,hit_mean)

#2
d_walk = data2011[(data2011['EVENT_CD']==14)]
d_walk_runner_on_first = d_walk[(d_walk['STATE']== '100_0')|(d_walk['STATE']== '100_1')|(d_walk['STATE']== '100_2')]
value_walk = d_walk_runner_on_first['RUNS_VALUE'].mean()

d_hitbypitch = data2011[(data2011['EVENT_CD']==16)]
d_hbp_runner_on_first = d_hitbypitch[(d_hitbypitch['STATE']== '100_0')|(d_hitbypitch['STATE']== '100_1')|(d_hitbypitch['STATE']== '100_2')]
value_hitbypitch = d_hbp_runner_on_first['RUNS_VALUE'].mean()

d_single = data2011[(data2011['EVENT_CD']==20)]
d_single_runner_on_first = d_single[(d_single['STATE']== '100_0')|(d_single['STATE']== '100_1')|(d_single['STATE']== '100_2')]
mean_single_runneronfirst = d_single_runner_on_first['RUNS_VALUE'].mean()

#3
#comparing rickie weeks and michael bourne
weeks = data2011[(data2011["BAT_ID"]=='weekr001')]
weeks = weeks[(weeks['BAT_EVENT_FL'] == 'T')]
#dataset of just state, new state, and run value
weeks_1 = weeks.loc[:,['STATE','NEW_STATE','RUNS_VALUE']]
weeks_total_value = weeks_1['RUNS_VALUE'].sum()
weeks_total_possible_value = weeks['RUNS_STATE'].sum()

bourne = data2011[(data2011["BAT_ID"]=='bourm001')]
bourne = bourne[(bourne['BAT_EVENT_FL'] == 'T')]
#dataset of just state, new state, and run value
bourne_1 = bourne.loc[:,['STATE','NEW_STATE','RUNS_VALUE']]
bourne_total_value = bourne_1['RUNS_VALUE'].sum()
bourne_total_possible_value = bourne['RUNS_STATE'].sum()


#4
#create probability of scoring a run matrix
data2011.columns
runs_scoring_plays = data2011[(data2011['RUNS_SCORED'] > 0)]
run_scoring_plays_states = runs_scoring_plays['STATE'].value_counts()
total_states = data2011['STATE'].value_counts()
states = pd.concat([total_states, run_scoring_plays_states], axis = 1, sort = True)
states['Probability_run']= states.iloc[:,1]/states.iloc[:,0]


#5
#Runner advancement with a single
#a
d_single = data2011[(data2011['EVENT_CD']==20)]
#b
single_state = d_single['STATE'].value_counts()
single_new_state = d_single['NEW_STATE'].value_counts()
#c
d_single_runner_on_first = d_single[(d_single['STATE']== '100_0')|(d_single['STATE']== '100_1')|(d_single['STATE']== '100_2')]
single_runner_on_first_state = d_single_runner_on_first['NEW_STATE'].value_counts()
lead_runner_to_second = single_runner_on_first_state['110_0'] + single_runner_on_first_state['110_1'] + single_runner_on_first_state['110_2']
lead_runner_to_third = single_runner_on_first_state['101_0'] + single_runner_on_first_state['101_1'] + single_runner_on_first_state['101_2'] + single_runner_on_first_state['011_1']