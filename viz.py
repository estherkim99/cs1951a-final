# Example Python program to draw a scatter plot

# for two columns of a multi-column DataFrame

import pandas as pd

import numpy as np

import matplotlib.pyplot as plot

import sqlite3 as sq


path_to_db = "./data/housing.db"

Conn = sq.connect(path_to_db)
#********************************** PER CITY ***************************************************
#query_2020 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2020%' GROUP BY city) SELECT a.ct, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
#query_2019 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2019%' GROUP BY city) SELECT a.ct, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
#query_2018 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2018%' GROUP BY city) SELECT a.ct, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
#query_2017 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2017%' GROUP BY city) SELECT a.ct, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
#query_2016 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2016%' GROUP BY city) SELECT a.ct, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
#query_2015 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2015%' GROUP BY city) SELECT a.ct, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)

#********************************** PER ZIPCODE ***************************************************
query_2020 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2020%' GROUP BY zipcode) SELECT a.ct, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
query_2019 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2019%' GROUP BY zipcode) SELECT a.ct, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
query_2018 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2018%' GROUP BY zipcode) SELECT a.ct, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
query_2017 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2017%' GROUP BY zipcode) SELECT a.ct, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
query_2016 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2016%' GROUP BY zipcode) SELECT a.ct, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
query_2015 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2015%' GROUP BY zipcode) SELECT a.ct, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)


# Load data into pandas DataFrame  
Df0 = pd.DataFrame(query_2020, columns=['ct', '2020']);
Df0.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
Df0.rename(columns = {'2020':'Avg Home Price'}, inplace = True) 
#print(Df0); 

Df1 = pd.DataFrame(query_2019, columns=['ct', '2019']);
Df1.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
Df1.rename(columns = {'2019':'Avg Home Price'}, inplace = True) 
#print(Df1);      

Df2 = pd.DataFrame(query_2018, columns=['ct', '2018']);
Df2.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
Df2.rename(columns = {'2018':'Avg Home Price'}, inplace = True) 
#print(Df2); 

Df3 = pd.DataFrame(query_2017, columns=['ct', '2017']);
Df3.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
Df3.rename(columns = {'2017':'Avg Home Price'}, inplace = True) 
#print(Df3); 

Df4 = pd.DataFrame(query_2016, columns=['ct', '2016']);
Df4.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
Df4.rename(columns = {'2016':'Avg Home Price'}, inplace = True) 
#print(Df4); 

Df5 = pd.DataFrame(query_2015, columns=['ct', '2015']);
Df5.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
Df5.rename(columns = {'2015':'Avg Home Price'}, inplace = True) 
#print(Df5); 
 

# Draw a scatter plot

Df0.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2020");
Df1.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2019");
Df2.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2018");
Df3.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2017");
Df4.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2016");
Df5.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2015");

plot.show(block=True);