import numpy as np
import pandas as pd
import sqlite3

path_hv = "./data/Zip_Zhvi_AllHomes.csv"
path_r = "./data/Zip_Zri_AllHomesPlusMultifamily.csv"
with open(path_hv) as f:
	df_hv = pd.read_csv(f)
print("Finished reading home values data!")

with open(path_r) as f:
	df_r = pd.read_csv(f)
print("Finished reading rental values data!")

# HOME VALUE DATA PROCESSING
'''
	Drop the following columns
	0. Region ID
	4. Metro
	5. CoutryName
	6. SizeRank
	7. - 231. All data from months before 2015
'''
cols_to_drop = [0] + list(range(4, 232))
df_hv.drop(df_hv.columns[cols_to_drop], axis=1, inplace=True)

'''
	Select rows corresponding to zip codes in relevant cities
'''

locations = ["BostonMA", "ChicagoIL", "San FranciscoCA", "New YorkNY", "NashvilleTN", "Los AngelesCA", "AustinTX", "SeattleWA", "DenverCO", "AshvilleNC"]
df_hv = df_hv.loc[(df_hv["City"] + df_hv["State"]).isin(locations)]

'''
        Clean data types
'''
df_hv[['City', 'State']] = df_hv[['City', 'State']].astype('str')
for i in range(3, len(df_hv.columns)) :
        df_hv[df_hv.columns[i]] = df_hv[df_hv.columns[i]].astype('float64')

print("Finished parsing home values data!")
print(df_hv.columns)
# print(df_hv.describe())
# print(df_hv.dtypes)

# RENTAL DATA PROCESSING
'''
	Drop the following columns
	0. Region ID
	4. Metro
	5. CoutryName
	6. SizeRank
	7. - 231. All data from months before 2015
'''
cols_to_drop = [0] + list(range(4, 59))
df_r.drop(df_r.columns[cols_to_drop], axis=1, inplace=True)

'''
	Select rows corresponding to zip codes in relevant cities
'''

locations = ["BostonMA", "ChicagoIL", "San FranciscoCA", "New YorkNY", "NashvilleTN", "Los AngelesCA", "AustinTX", "SeattleWA", "DenverCO", "AshvilleNC"]
df_r = df_r.loc[(df_r["City"] + df_r["State"]).isin(locations)]

'''
        Clean data types
'''
df_r[['City', 'State']] = df_r[['City', 'State']].astype('str')
for i in range(3, len(df_r.columns)) :
        df_r[df_r.columns[i]] = df_r[df_r.columns[i]].astype('float64')

print("Finished parsing rental values data!")
print(df_r.columns)
# print(df_hv.describe())
# print(df_r.dtypes)



'''
        INSERTING TABLES INTO DATABASE
'''
# Create connection to database
conn = sqlite3.connect('./data/test.db')
c = conn.cursor()

df_hv.to_sql("zillow_zhvi", conn, if_exists="replace")
df_r.to_sql("zillow_zri", conn, if_exists="replace")

conn.close()