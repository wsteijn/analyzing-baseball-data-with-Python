# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:08:19 2021

@author: Will
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import math

#Justin Verlander PITCHFX data
verlander = pd.read_csv('Verlander.csv')
#random sample of rows from the verlander data frame
random_subset = verlander.sample(n=20)
print(random_subset)

#histogram of the speed of verlanders pitches
sns.displot(verlander, x='speed')
#density plot of the speed of verlanders pitches
sns.displot(verlander, x='speed', kind = 'kde')

#make individual dataframes for each pitch type
pitch_type = verlander['pitch_type'].unique()
output_dfs = {p: verlander[verlander['pitch_type']==p] for p in pitch_type}
verlander_FF = output_dfs[pitch_type[0]]
verlander_CU = output_dfs[pitch_type[1]]
verlander_CH = output_dfs[pitch_type[2]]
verlander_SL = output_dfs[pitch_type[3]]
verlander_FT = output_dfs[pitch_type[4]]

#density plot for each pitch type
sns.displot(verlander_FF, x='speed', kind = 'kde')
sns.displot(verlander_FT, x='speed', kind = 'kde')
sns.displot(verlander_SL, x='speed', kind = 'kde')
sns.displot(verlander_CU, x='speed', kind = 'kde')
sns.displot(verlander_CH, x='speed', kind = 'kde');

#panel of density plots for each pitch type
g = sns.FacetGrid(verlander, row = 'pitch_type', height = 2, aspect= 4,)
g.map(sns.kdeplot, 'speed')

#superimpose density plots on top of each other
sns.displot(verlander, x = 'speed', hue = 'pitch_type', kind = 'kde')

#scatterplots and dot plots
import datetime
#function to turn gamedate column into an integer value for the number of day in that year
def get_day(gamedate):
    day_number = datetime.datetime.strptime(gamedate, '%Y-%m-%d')
    day_number = int(day_number.strftime('%j'))
    return day_number
#create gameday column = day of the year as a interger from 1 to 365
verlander_FF['gameDay']=verlander['gamedate'].apply(lambda x: get_day(x))
#create dataframe that has the average speeds of verlander's fastball by the day of the year and the season
dailySpeed = verlander_FF.groupby(['gameDay','season'], as_index=False)['speed'].mean()
#create regression plot of average speed of verlander's fastball over the course of each season
g = sns.FacetGrid(dailySpeed, col='season', col_wrap = 2, height = 4)
g.map(sns.regplot,'gameDay','speed', lowess=True)

#dataframe of only fastballs and changeups
speedFC = pd.concat([verlander_FF, verlander_CH])
#average speeds by season and pitch type
avgspeedFC = speedFC.groupby(['season','pitch_type'], as_index = False)['speed'].mean()
avgspeedFC['speed'] = avgspeedFC['speed'].round(1)
g = sns.stripplot(x = 'speed', y = 'season', hue = 'pitch_type', data=avgspeedFC)
#g.set(xlim = (80, 100), xlabel = 'Speed')
g.set(ylim=(2008,2013),yticks=[2009,2010,2011,2012])
g.xaxis.grid(False)
g.yaxis.grid(True)

#the panel function
avgSpeed = verlander_FF.groupby(['pitches','season'], as_index=False)['speed'].mean()
avgSpeedComb = verlander_FF['speed'].mean()
season = verlander_FF['season'].unique()
plt.style.use('ggplot')
g = sns.FacetGrid(avgSpeed, col = 'season', col_wrap = 2, height= 4,)
g.map(sns.scatterplot,'pitches','speed')
g.map(plt.axhline, y= avgSpeedComb, ls= '--',color='black')
g.map(plt.axvline, x = 100, ls = '-', color = 'black')
for ax, title in zip(g.axes.flat, season):
    ax.set_title(title)
    ax.text(5,96.5,'Average speed', fontsize=12)
plt.show()

#building a graph, step by step
NoHit = verlander[verlander['gamedate'] == '2011-05-07']
#add kzone
top_in_Kzone = [-.95, 3.5]
bot_in_Kzone = [-.95, 1.6]
top_out_Kzone = [.95, 3.5]
bot_out_Kzone = [0.95, 1,6]
#final plot with kzone 
plt.style.use('ggplot')
g = sns.FacetGrid(NoHit, col='batter_hand', col_wrap = 2, height = 4, col_order = ['L','R'], aspect = 1)
g.map(sns.scatterplot,'px','pz', style = 'pitch_type', hue = 'pitch_type', data= NoHit, legend = 'full')
def const_line_1(*args, **kwargs):
    x = np.arange(-.95, .95, .001)
    y = [3.5]*x.size
    plt.plot(x,y, C='k')
def const_line_2(*args, **kwargs):
    x = np.arange(-.95, .95, .001)
    y = [1.6]*x.size
    plt.plot(x,y, C='k')
def const_line_3(*args, **kwargs):
    y = np.arange(1.6, 3.5, .001)
    x = [0.95]*y.size
    plt.plot(x,y, C='k')
def const_line_4(*args, **kwargs):
    y = np.arange(1.6, 3.5, .001)
    x = [-0.95]*y.size
    plt.plot(x,y, C='k')
g.map(const_line_1)
g.map(const_line_2)
g.map(const_line_3)
g.map(const_line_4)
g.add_legend()
g.set(xlim = (-2.2,2.2), ylim = (0,5), xlabel = 'Horizontal location', ylabel = 'Vertical location')
g.fig.set_figwidth(9)
g.fig.set_figheight(5)
g.tight_layout()
plt.show()

#The cabrera dataset

#Justin Verlander PITCHFX data
cabrera = pd.read_csv('Cabrera.csv')
#random sample of rows from the cabrera data frame
random_subset = cabrera.sample(n=20)
print(random_subset)

#subset of just batted balls for cabrera
cabrera.dropna(subset=['hitx'], inplace = True)

g = sns.scatterplot(x= cabrera['hitx'], y = cabrera['hity'], hue = cabrera['hit_outcome'])
plt.Axes.set_aspect(g, aspect = 'equal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#bases
homeplate = [0,0]
firstbase = [(90/2**.5),(90/2**.5)]
secondbase = [0, 2 * (90/2**.5)]
thirdbase = [-(90/2**.5),(90/2**.5)]

plt.style.use('ggplot')
g = sns.FacetGrid(cabrera, col='season', col_wrap = 2, height = 4, aspect = 1)
g.map(sns.scatterplot,'hitx','hity', hue = 'hit_outcome', data= cabrera, legend = 'full')
def const_line_1(*args, **kwargs):
    x1_values = [homeplate[0], 300]
    y1_values = [homeplate[1], 300]
    plt.plot(x1_values,y1_values, C='k')
def const_line_2(*args, **kwargs):
    x1_values = [firstbase[0], secondbase[0]]
    y1_values = [firstbase[1], secondbase[1]]
    plt.plot(x1_values,y1_values, C='k')
def const_line_3(*args, **kwargs):
    x1_values = [secondbase[0], thirdbase[0]]
    y1_values = [secondbase[1], thirdbase[1]]
    plt.plot(x1_values,y1_values, C='k')
def const_line_4(*args, **kwargs):
    x1_values = [homeplate[0], -300]
    y1_values = [homeplate[1], 300]
    plt.plot(x1_values,y1_values, C='k')    
g.map(const_line_1)
g.map(const_line_2)
g.map(const_line_3)
g.map(const_line_4)
g.add_legend()
g.fig.set_figwidth(10)
g.fig.set_figheight(9)
plt.show()

#Combining information
cabreraStretch = cabrera[cabrera['gamedate']> '2012-08-31']
#make speed categories
def speed_category(x):
    if x< 80:
        return 70
    if (x >= 80) & (x <90):
        return 80
    if (x >=90) & (x< 100):
        return 90
    else:
        return 100
cabreraStretch['speed_category']=cabreraStretch['speed'].apply(lambda x: speed_category(x))

plt.style.use('ggplot')
g = sns.scatterplot(x='hitx',y='hity', style = 'hit_outcome', hue = 'pitch_type', size = 'speed_category', sizes=(40, 400), data= cabreraStretch, legend = 'full')
plt.plot([homeplate[0],300],[homeplate[1],300], C='k')
plt.plot([firstbase[0],secondbase[0]],[firstbase[1],secondbase[1]], C='k')
plt.plot([secondbase[0],thirdbase[0]],[secondbase[1],thirdbase[1]], C='k')
plt.plot([homeplate[0],-300],[homeplate[1],300], C='k')
plt.Axes.set_aspect(g, aspect = 'equal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#verlander fastballs per season with smoothing line
avgSpeed = verlander_FF.groupby(['pitches','season'], as_index=False)['speed'].mean()
avgSpeedComb = verlander_FF['speed'].mean()
season = verlander_FF['season'].unique()

verlander_subset = verlander_FF.sample(n=1000)


plt.style.use('ggplot')
g = sns.FacetGrid(verlander_FF, col = 'season', col_wrap = 2, height= 4,)
g.map(sns.regplot,x ='pitches',y ='speed',data = verlander_FF, scatter_kws={"color": "red"}, line_kws={"color": "black"}, lowess = True, ci=68, scatter = True)
g.map(plt.axhline, y= avgSpeedComb, ls= '--',color='black')
g.map(plt.axvline, x = 100, ls = '-', color = 'black')
for ax, title in zip(g.axes.flat, season):
    ax.set_title(title)
    ax.text(5,96.5,'Average speed', fontsize=12)
plt.show()

g = sns.lmplot(x = 'pitches', y = 'speed', hue = 'season', data = verlander_FF, palette = 'Set1', scatter = False)

g = sns.lmplot(x = 'pitches', y = 'speed', col = 'season', data = verlander_FF, palette = 'Set1', col_wrap = 2, scatter = False)
g.map(plt.axhline, y= avgSpeedComb, ls= '--',color='black')
g.map(plt.axvline, x = 100, ls = '-', color = 'black')
for ax, title in zip(g.axes.flat, season):
    ax.set_title(title)
    ax.text(5,96.5,'Average speed', fontsize=12)
plt.show()

#hexagonal bins having the bins colored according to the number of data points they contain
sns.jointplot(x = 'px', y = 'pz', data = verlander_FF, kind = 'hex')

fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
ax = axs[0]
hb = ax.hexbin(verlander_FF['px'], verlander_FF['pz'], gridsize=50, cmap='inferno')
#ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("Hexagon binning")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('counts')

ax = axs[1]
hb = ax.hexbin(verlander_CU['px'], verlander_CU['pz'], gridsize=50, cmap='inferno')
#ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("With a log color scale")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')

plt.show()


def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)    

g = sns.FacetGrid(verlander_FF, col="batter_hand")
h= g.map(hexbin, "px", "pz")
def const_line_1(*args, **kwargs):
    x = np.arange(-.95, .95, .001)
    y = [3.5]*x.size
    plt.plot(x,y, C='k')
def const_line_2(*args, **kwargs):
    x = np.arange(-.95, .95, .001)
    y = [1.6]*x.size
    plt.plot(x,y, C='k')
def const_line_3(*args, **kwargs):
    y = np.arange(1.6, 3.5, .001)
    x = [0.95]*y.size
    plt.plot(x,y, C='k')
def const_line_4(*args, **kwargs):
    y = np.arange(1.6, 3.5, .001)
    x = [-0.95]*y.size
    plt.plot(x,y, C='k')
g.map(const_line_1)
g.map(const_line_2)
g.map(const_line_3)
g.map(const_line_4)
g.set(xlim = (-2.2,2.2), ylim = (0.5,5), xlabel = 'Horizontal location', ylabel = 'Vertical location')
g.fig.set_figwidth(9)
g.fig.set_figheight(5)
plt.colorbar()

ax = g.axes[0]









