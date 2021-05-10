# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 02:56:50 2021

@author: Will
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import math
import re

#Mussina ops allowed by count state
balls = list(range(0,4))*3
strikes = list(range(0,3))*4
strikes.sort()
#Mussina OPS values
value = list([100, 118, 157, 207, 72, 82, 114, 171, 30, 38, 64, 122])
#create dataframe of balls, strikes, and ops value
mussina = pd.DataFrame(data = {'balls': balls, 'strikes': strikes, 'value': value})
#create dataframe where the count is x-y axis
mussina1 = pd.pivot_table(mussina, values = 'value', index = ['balls'], columns = 'strikes')
#heatmap of ops by count
ax = sns.heatmap(mussina1, annot = True)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!

#Pitch sequences on Retrosheet
#functions for string manipulation

#obtain number of pitches delivered in a Retrosheet pitch sequence
len('BBSBFFFX')
#find which sequences have a specific character or pattern
sequences = ['BBX','C11BBC1S','1BX']
[bool(re.search('^[BC]',i)) for i in sequences]
[bool(re.search('11',i)) for i in sequences]
['BB' in i for i in sequences]
#sub one character for another - remove pickoff attempts to first from the pitch sequence
sequences = pd.Series(['BBX','C11BBC1S','1X'])
sequences_no_pickoffs = sequences.replace('1', "", regex=True)

#Finding plate appearances going through a given count
#load play by play retrosheet data for 2011 season
pbp2011 = pd.read_csv('all2011.csv', header = None)
fields = pd.read_csv('fields.csv')
pbp2011.columns = fields['Header']
#create new variable pseq of pitch sequences which removes non-pitch symbols
#replace nan with empty space
pbp2011['PITCH_SEQ_TX'] = pbp2011['PITCH_SEQ_TX'].replace(np.nan, "", regex = True)
#alternatively - remove rows with NA in pitch sequence
pbp2011.dropna(subset = ["PITCH_SEQ_TX"], inplace = True)
#create new variables
pbp2011['pseq'] = pbp2011['PITCH_SEQ_TX'].replace("[.>123N+*]", "", regex=True)
#create new variable for plate appearances that go through a 1-0 count
pbp2011['c10'] = pbp2011.apply(lambda x: bool(re.search("^[BIPV]", x['pseq'])), axis = 1)
#create new variable for plate appearances that go through a 0-1 count
pbp2011['c01'] = pbp2011.apply(lambda x: bool(re.search("^[CFKLMOQRST]", x['pseq'])), axis =1)
pbp2011[['PITCH_SEQ_TX', 'c10','c01']][0:10]
pbp2011['c20'] = pbp2011.apply(lambda x: bool(re.search("^[BIPV]{2}", x['pseq'])), axis = 1)
pbp2011['c30'] = pbp2011.apply(lambda x: bool(re.search("^[BIPV]{3}", x['pseq'])), axis = 1)
pbp2011['c02'] = pbp2011.apply(lambda x: bool(re.search("^[CFKLMOQRST]{2}", x['pseq'])), axis = 1)
pbp2011['c11'] = pbp2011.apply(lambda x: bool(re.search("^([CFKLMOQRST][BIPV]|[BIPV][CFKLMOQRST])", x['pseq'])), axis = 1)
pbp2011['c21'] = pbp2011.apply(lambda x: bool(re.search("^([CFKLMOQRST][BIPV][BIPV]|[BIPV][CFKLMOQRST][BIPV]|[BIPV][BIPV][CFKLMOQRST])", x['pseq'])), axis = 1)
pbp2011['c31'] = pbp2011.apply(lambda x: bool(re.search("^([CFKLMOQRST][BIPV][BIPV][BIPV]|[BIPV][CFKLMOQRST][BIPV][BIPV]|[BIPV][BIPV][CFKLMOQRST][BIPV]|[BIPV][BIPV][BIPV][CFKLMOQRST])", x['pseq'])), axis = 1)
pbp2011['c12'] = pbp2011.apply(lambda x: bool(re.search("^([CFKLMOQRST][CFKLMOQRST][FR]*[BIPV]|[BIPV][CFKLMOQRST][CFKLMOQRST]|[CFKLMOQRST][BIPV][CFKLMOQRST])", x['pseq'])), axis = 1)
pbp2011['c22'] = pbp2011.apply(lambda x: bool(re.search("^([CFKLMOQRST][CFKLMOQRST][FR]*[BIPV][FR]*[BIPV]|[BIPV][BIPV][CFKLMOQRST][CFKLMOQRST]|[BIPV][CFKLMOQRST][BIPV][CFKLMOQRST]|[BIPV][CFKLMOQRST][CFKLMOQRST][FR]*[BIPV]|[CFKLMOQRST][BIPV][CFKLMOQRST][FR]*[BIPV]|[CFKLMOQRST][BIPV][BIPV][CFKLMOQRST])", x['pseq'])), axis = 1)

pbp2011['c32_1'] = pbp2011.apply(lambda x: bool(re.search("^[CFKLMOQRST]*[BIPV][CFKLMOQRST]*[BIPV][CFKLMOQRST]*[BIPV]", x['pseq'])), axis = 1)
pbp2011['c32_2'] = pbp2011.apply(lambda x: bool(re.search("^[BIPV]*[CFKLMOQRST][BIPV]*[CFKLMOQRST]", x['pseq'])), axis = 1)

pbp2011rc_1 = pd.read_csv('pbp11rc.csv')
pbp2011rc_1.head()
pbp2011rc_1.columns.values
sum(pbp2011rc_1['c32'])
sum(pbp2011['c32_1'])
sum(pbp2011['c32_2'])
pbp2011['c32'] = pbp2011.apply(lambda x: bool(x['c32_1'] == True & x['c32_2'] == True), axis = 1)
sum(pbp2011['c32'])

sum(pbp2011rc_1['c22'])
sum(pbp2011['c22'])

