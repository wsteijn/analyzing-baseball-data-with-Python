# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:05:43 2020

@author: Will
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


#2.3.2 Vectors: defining and calculations
W = np.array([8, 21, 15, 21, 21, 22, 14])
L = np.array([5, 10, 12, 14, 17, 14, 19])

win_pct = np.array(100*W/(W+L))

year = np.linspace(1946,1952, num=7)
year1 = np.array(list(range(1946, 1953, 1)))

Age = year1 - 1921

#plot - execute all three lines together
plt.xlabel("Age")
plt.ylabel("Win_pct")
plt.plot(Age,win_pct,'o', color='black')

#built in numpy functions
np.mean(win_pct)
np.sort(W)
len(L)
np.std(win_pct)
np.cumsum(W)

#Vector index
import operator
#extract first, second, and fifth entries in vector W
extract_elements = operator.itemgetter(0,1,4)
extract_elements(W)

#extract first four entries of W
W[0:4]

#remove first and sixth entries of W
W=W.tolist()
W.pop(0)
W.pop(4)
W

#Values of win.pct >60
logical_vector = win_pct > 60
#can only be done between arrays, not list and an array (so change W back to an array)
(W > 20) & (win_pct > 60)
#what year was the highest win pct?
win_pct == np.max(win_pct)
year[win_pct == np.max(win_pct)]

#which years did he have more than 30 decisions
year[W + L > 30]


#using sapply
#who hit the most homeruns in the 60s?
#read dataset
Batting = pd.read_csv("Batting.csv")
Batting.head()
#subset of batting dataset for only the 60s
Batting_60 = Batting[(Batting["yearID"]>=1960) & (Batting["yearID"]<=1969)]
#function for computing homeruns given a player id
def compute_hr(pid):
    d = Batting_60[(Batting_60['playerID']==pid)]
    return d['HR'].sum()
#array of all the individual player IDs
players = pd.unique(Batting_60['playerID'])
#apply the fucntion to each individual player id in the array
S = pd.Series(players).apply(lambda x: compute_hr(x))
#create data frame to combine players vector and S vector
R = pd.DataFrame({'Players':players, 'HRs':S}, columns = ['Players', 'HRs'])
#order by HRs
R = R.sort_values("HRs", ascending=False)
R.head()

#collect the career at-bats, homeruns, and Ks for all players with at least 5000 career at bats
#group by player and sum their stats to form a df with each players career stats
dataframe_AB = Batting.groupby(by = ["playerID"]).sum()
#subset dataset for only players with above 5000 ABs
dataframe_AB = dataframe_AB[(dataframe_AB["AB"]>=5000)]
#subset data frame for just the stats of interest - the playerID is the index of the dataframe, not a column
col_names = ["AB", "HR", "SO"]
d_5000 = dataframe_AB[col_names]

HR_per_AB = pd.array(d_5000["HR"]/d_5000["AB"])
SO_per_AB = pd.array(d_5000["SO"]/d_5000["AB"])

plt.xlabel("HR/AB")
plt.ylabel("SO/AB")
plot = sns.regplot(HR_per_AB, SO_per_AB, lowess = True, scatter_kws={"color": "black"}, line_kws={"color": "red"})

#view a specific row by index
dataframe_AB.loc["aaronha01"]
#view column names
dataframe_AB.columns
#view dataframe head
d_5000.head()

#exercises
#1
Player = np.array(["Rickey Henderson", "Lou Brock", "Ty Cobb", "Eddie Collins", "Max Carey", "Joe Morgan", "Luis Aparicio", "Paul Molitor", "Roberto Alomar"])
SB = np.array([1406, 938, 897, 741, 738, 689, 506, 504, 474])
CS = np.array([335, 307, 212, 195, 109, 162, 136, 131, 114])
G = np.array([3081, 2616, 3034, 2826, 2476, 2649, 2599, 2683, 2379])
#calculate stolen base attempts
SB_attempt = SB + CS
#calculate success rate
Success_Rate = SB/ SB_attempt
#calculate stolen bases per game
SB_Game = SB/G
#put in data frame
stolen_base_df = pd.DataFrame({'Player':Player, 'SB_Game':SB_Game, 'Success_Rate':Success_Rate}, columns = ['Player', 'SB_Game', 'Success_Rate'])
stolen_base_df.head()
#create scatter plot with the player names as labels for the data points
ax = stolen_base_df.plot(x = 'SB_Game', y= 'Success_Rate', kind='scatter')
stolen_base_df[['SB_Game','Success_Rate','Player']].apply(lambda x: ax.text(*x), axis =1)

#2
#list of outcomes
outcomes = ['Single', 'Out', 'Out', 'Single', 'Out', 'Double', 'Out', 'Walk', 'Out', 'Single']
#import a counter
from collections import Counter
outcome_frequency = Counter(outcomes)

#3
W = pd.array([373, 354, 364, 417, 355, 373, 361, 363, 511])
L = pd.array([208, 184, 310, 279, 227, 188, 208, 245, 316])
Name = ['Pete Alexander', 'Roger Clemens', 'Pud Galvin', 'Walter Johnson', 'Greg Maddux', 'Christy Mathewson', 'Kid Nichols', 'Warren Spahn', 'Cy Young']
#compute winning percentage
Win_Pct = W/(W+L)
#create dataframe with Name, W, L, Win_Pct
Win_350 = pd.DataFrame({'Name':Name, 'W':W,'L':L, 'Win Percentage':Win_Pct}, columns = ['Name', 'W','L','Win Percentage'])
Win_350.head()
#sort by win_pct
Win_350.sort_values('Win Percentage', ascending=False)

#4
SO = pd.array([2198, 4672, 1806, 3509, 3371, 2502, 1868, 2583, 2803])
BB = pd.array([951, 1580, 745, 1363, 999, 844, 1268, 1434, 1217])
#compute K to BB ratio
SO_BB_Ration = SO/BB
#create dataframe with names, Ks, BBs, and K to BB ratio
SO_BB = pd.DataFrame({'Name':Name, 'SO':SO,'BB':BB, 'K to BB Percentage':SO_BB_Ration}, columns = ['Name', 'SO','BB','K to BB Percentage'])
#use subset function to find the pitcers with a K-bb ratio over 2.8
SO_BB = SO_BB[(SO_BB["K to BB Percentage"]>2.8)]
#sort dataframe by number of walks
SO_BB.sort_values('BB', ascending=False)

#5
Pitching = pd.read_csv("Pitching.csv")
Pitching.head()
#compute cumulative Ks, BBs, mid career year, total innings pitched
#create array of all the unique player IDs in the pitching.csv dataset
pitchers = pd.unique(Pitching['playerID'])
#function to compute the career stats for each playerid
def compute_stats(pid):
    d = Pitching[(Pitching['playerID']==pid)]
    return pd.array([d['SO'].sum(), d['BB'].sum(), d['IPouts'].sum(),d['yearID'].median()])
#apply the compute stats function to each player id
S = pd.Series(pitchers).apply(lambda x: compute_stats(x))
#create a dataframe of the  stats
career_pitching = pd.DataFrame(S.values.tolist(), index=S.index)
#define column names
career_pitching.columns= ['SO', 'BB', 'IPouts', 'medYear']
#insert playerID array as the first column in the dataframe
career_pitching.insert(loc=0, column='playerID', value=pitchers)
career_pitching.head()
#check stats function
compute_stats('bechtge01')
#make new dataframe for pitchers with at least 10,000 career IPouts
career_10000 = career_pitching[(career_pitching["IPouts"]>10000)]
#construct a scatterplot of mid career year and ratio of Ks to BB
K_BB = career_10000['SO']/career_10000['BB']
career_10000.insert(loc=1, column='K_BB_ratio', value=K_BB)
career_10000.head()
plt.xlabel("medYear")
plt.ylabel("K/BB ratio")
plot = sns.regplot(career_10000['medYear'], career_10000['K_BB_ratio'], lowess = True, scatter_kws={"color": "black"}, line_kws={"color": "red"})

career_10000['K_BB_ratio'].max()
career_10000.sort_values('K_BB_ratio', ascending=False)
career_10000.head()

