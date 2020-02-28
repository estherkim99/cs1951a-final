import numpy as np
import pandas as pd
import sqlite3

path = "./data/Zip_Zhvi_AllHomes.csv"
with open(path) as f:
	df = pd.read_csv(f)
print("Finished reading data!")

'''
	Drop the following columns
	0. Region ID
	4. Metro
	5. CoutryName
	6. SizeRank
	7. - 231. All data from months before 2015
'''
cols_to_drop = [0] + list(range(4, 232))
df.drop(df.columns[cols_to_drop], axis=1, inplace=True)

'''
	Select rows corresponding to zip codes in relevant cities
'''
locations = ["BostonMA", "ChicagoIL", "San FranciscoCA", "New YorkNY", "NashvilleTN", "Los AngelesCA", "AustinTX", "SeattleWA", "DenverCO", "AshvilleNC"]
df = df.loc[(df["City"] + df["State"]).isin(locations)]

# df = df.groupby(["City"])

print(df.columns)
print(df.describe())

# Create connection to database
conn = sqlite3.connect('./data/test.db')
c = conn.cursor()

# Delete tables if they exist
c.execute('DROP TABLE IF EXISTS "zillow_zhvi";')

# Create tables in the database and add data to it. REMEMBER TO COMMIT
'''
c.execute("""CREATE TABLE zillow_zhvi(
        symbol text not null, 
        name text, 
        location text);""")
'RegionName', 'City', 'State', '2015-01', '2015-02', '2015-03',
       '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09',
       '2015-10', '2015-11', '2015-12', '2016-01', '2016-02', '2016-03',
       '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09',
       '2016-10', '2016-11', '2016-12', '2017-01', '2017-02', '2017-03',
       '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09',
       '2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03',
       '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09',
       '2018-10', '2018-11', '2018-12', '2019-01', '2019-02', '2019-03',
       '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09',
       '2019-10', '2019-11', '2019-12', '2020-01'
'''
