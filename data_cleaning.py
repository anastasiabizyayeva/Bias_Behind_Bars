# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:42:53 2020

@author: Anastasia
"""
import pandas as pd

#Read in data
df = pd.read_csv('Globe_Data.csv')

#Drop redacted column 
df = df.drop(['JUDGE'], axis=1)

#Drop records without a race 
df.dropna(subset = ["RACE"], inplace=True)

#Correct typo
df.rename(columns={'INSTUTUTIONAL SECURITY LEVEL': 'INSTITUTIONAL SECURITY LEVEL'}, inplace=True)

#Convert year to readable format 
df['FISCAL YEAR'] = df['FISCAL YEAR'].apply(lambda x: x[2:])
df['FISCAL YEAR'] = df['FISCAL YEAR'].apply(lambda x: x[:2] + '-' + x[2:])

#Condense racial categories to visible racial groups
race_cat = {('Arab', 'Arab/West Asian', 'Asian-West'):'Middle Eastern/ West Asian', ('Asi-E/Southeast', 'Asiatic', 'Chinese', 'Filipino', 'Japanese', 'Korean', 'S. E. Asian'):'East Asian', ('Asian-South', 'East Indian', 'South Asian'):'South Asian', ('British Isles', 'Euro.-Eastern', 'Euro.-Northern', 'Euro.-Southern', 'Euro.-Western', 'European French', 'White'):'White', ('Black', 'Caribbean', 'Sub-Sahara Afri'):'Black', ('Hispanic', 'Latin American'):'Hispanic', ('Inuit', 'Metis', 'North American'):'Indigenous', ('Other', 'Unable Specify', 'Unknown'):'Other'}

race_dict = {}
for k, v in race_cat.items():
    for key in k:
        race_dict[key] = v
        
race_dict['Oceania'] = 'Oceania'
race_dict['Multirac/Ethnic'] = 'Multi-Ethic'
        
df['RACIAL CATEGORY'] = df['RACE'].map(race_dict)

#Drop records with sentence length less than 0 
df = df[df['AGGREGATE SENTENCE LENGTH'] >= 0.0]

#Create column that converts sentence length from days into years
df['SENTENCE LENGTH (YEARS)'] = df['AGGREGATE SENTENCE LENGTH'].apply(lambda x: round(x/365, 2))

#Export to CSV
df.to_csv('cleaned_data.csv')