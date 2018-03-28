# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 09:34:42 2017

@author: aszewczyk
"""

import pandas as pd

raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 
        'sex': ['male', 'female', 'male', 'female', 'female']}

df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'sex'])

df.dtypes

df_sex = pd.get_dummies(df['sex'], drop_first = True)

df_new = pd.concat([df, df_sex], axis=1)
#df_new = df.join(df_sex) #alternatively


df_new = df_new.drop(['sex'], axis=1)
