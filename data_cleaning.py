# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:42:53 2020

@author: Anastasia
"""
import pandas as pd
import numpy as np

df = pd.read_csv('Globe_Data.csv')

df = df.drop(['JUDGE'], axis=1)

df.dropna(subset = ["RACE"], inplace=True)

df.rename(columns={'INSTUTUTIONAL SECURITY LEVEL': 'INSTITUTIONAL SECURITY LEVEL'}, inplace=True)


df['FISCAL YEAR'] = df['FISCAL YEAR'].apply(lambda x: x[2:])
df['FISCAL YEAR'] = df['FISCAL YEAR'].apply(lambda x: x[:2] + '-' + x[2:])

race_cat = {('Arab', 'Arab/West Asian', 'Asian-West'):'Middle Eastern/ West Asian', ('Asi-E/Southeast', 'Asiatic', 'Chinese', 'Filipino', 'Japanese', 'Korean', 'S. E. Asian'):'East Asian', ('Asian-South', 'East Indian', 'South Asian'):'South Asian', ('British Isles', 'Euro.-Eastern', 'Euro.-Northern', 'Euro.-Southern', 'Euro.-Western', 'European French', 'White'):'White', ('Black', 'Caribbean', 'Sub-Sahara Afri'):'Black', ('Hispanic', 'Latin American'):'Hispanic', ('Inuit', 'Metis', 'North American'):'Indigenous', ('Other', 'Unable Specify', 'Unknown'):'Other'}

race_dict = {}
for k, v in race_cat.items():
    for key in k:
        race_dict[key] = v
        
race_dict['Oceania'] = 'Oceania'
race_dict['Multirac/Ethnic'] = 'Multi-Ethic'
        
df['RACIAL GROUPS'] = df['RACE'].map(race_dict)

print(df.groupby('RACE')['SENTENCE ID'].nunique())
print(df['OFFENDER NUMBER'].head())
print(df['SENTENCE ID'].nunique())
print(df['SENTENCE ID'].count())