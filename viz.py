import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3 as sq
from scipy import stats
import statsmodels.api as sm

path_to_db = "./data/housing.db"

#
def cities_by_year():
	Conn = sq.connect(path_to_db)

	df = pd.read_sql_query('''SELECT city, AVG("2015"), AVG("2016"), AVG("2017"), AVG("2018"), AVG("2019"), AVG("2020") FROM zillow_zhvi_yearavg GROUP BY city;''', Conn)
	df.set_index('city',inplace=True)
	df = df.transpose()
	# print(df.describe)
	# print(df.columns)

	fig = plt.figure()
	ax = plt.subplot(111)

	X = ['2015', '2016', '2017', '2018', '2019', '2020']
	ax.plot(X, 'Asheville', data=df)
	ax.plot(X, 'Austin', data=df)
	ax.plot(X, 'Boston', data=df)
	ax.plot(X, 'Chicago', data=df)
	ax.plot(X, 'Denver', data=df)
	ax.plot(X, 'Los Angeles', data=df)
	ax.plot(X, 'Nashville', data=df)
	ax.plot(X, 'New York', data=df)
	ax.plot(X, 'San Francisco', data=df)
	ax.plot(X, 'Seattle', data=df)

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	plt.xlabel('Year')
	plt.ylabel('Average Home Price')
	plt.title('Average Home Price by Year in Each City')
	plt.show()

# cities_by_year()


def zipcode_scatterplots():
	Conn = sq.connect(path_to_db)
	#********************************** PER CITY ***************************************************
	#query_2020 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2020%' GROUP BY city) SELECT a.ct, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	#query_2019 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2019%' GROUP BY city) SELECT a.ct, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	#query_2018 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2018%' GROUP BY city) SELECT a.ct, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	#query_2017 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2017%' GROUP BY city) SELECT a.ct, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	#query_2016 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2016%' GROUP BY city) SELECT a.ct, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	#query_2015 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2015%' GROUP BY city) SELECT a.ct, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)

	#********************************** PER ZIPCODE ***************************************************
	# query_2020 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2020%' and cityname = 'New York' GROUP BY zipcode) SELECT a.ct, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
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


	Df1 = pd.DataFrame(query_2019, columns=['ct', '2019']);
	Df1.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df1.rename(columns = {'2019':'Avg Home Price'}, inplace = True)

	Df2 = pd.DataFrame(query_2018, columns=['ct', '2018']);
	Df2.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df2.rename(columns = {'2018':'Avg Home Price'}, inplace = True)

	Df3 = pd.DataFrame(query_2017, columns=['ct', '2017']);
	Df3.dropna(subset = ["2017"], inplace=True)
	Df3.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df3.rename(columns = {'2017':'Avg Home Price'}, inplace = True)
	#print(Df3)

	Df4 = pd.DataFrame(query_2016, columns=['ct', '2016']);
	Df4.dropna(subset = ["2016"], inplace=True)
	Df4.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df4.rename(columns = {'2016':'Avg Home Price'}, inplace = True)
	#print(Df4)

	Df5 = pd.DataFrame(query_2015, columns=['ct', '2015']);
	Df5.dropna(subset = ["2015"], inplace=True)
	Df5.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df5.rename(columns = {'2015':'Avg Home Price'}, inplace = True)

	# Removes outliers (defined as points that are more than 3 standard deviations away)
	Df0 = Df0[(np.abs(stats.zscore(Df0)) < 3).all(axis=1)]
	Df1 = Df1[(np.abs(stats.zscore(Df1)) < 3).all(axis=1)]
	Df2 = Df2[(np.abs(stats.zscore(Df2)) < 3).all(axis=1)]
	Df3 = Df3[(np.abs(stats.zscore(Df3)) < 3).all(axis=1)]
	Df4 = Df4[(np.abs(stats.zscore(Df4)) < 3).all(axis=1)]
	Df5 = Df5[(np.abs(stats.zscore(Df5)) < 3).all(axis=1)]

	# Calculate the correlation coefficient and p-value using Pearson's coefficient by YEAR
	X0 = Df0['Avg Home Price'].values
	Y0 = Df0['Number of Airbnbs'].values
	print("2020: " + str(stats.pearsonr(X0, Y0)))

	X1 = Df1['Avg Home Price'].values
	Y1 = Df1['Number of Airbnbs'].values
	print("2019: " + str(stats.pearsonr(X1, Y1)))

	X2 = Df2['Avg Home Price'].values
	Y2 = Df2['Number of Airbnbs'].values
	print("2018: " + str(stats.pearsonr(X2, Y2)))

	X3 = Df3['Avg Home Price'].values
	Y3 = Df3['Number of Airbnbs'].values
	print("2017: " + str(stats.pearsonr(X3, Y3)))

	X4 = Df4['Avg Home Price'].values
	Y4 = Df4['Number of Airbnbs'].values
	print("2016: " + str(stats.pearsonr(X4, Y4)))

	X5 = Df5['Avg Home Price'].values
	Y5 = Df5['Number of Airbnbs'].values
	print("2015: " + str(stats.pearsonr(X5, Y5)))

	'''
	X0 = Df0['Avg Home Price']
	Y0 = Df0['Number of Airbnbs']
	model0 = sm.OLS(Y0, X0).fit()
	predictions = model.predict(X0) # make the predictions by the model
	# Print out the statistics
	print(model.summary())
	'''

	# Draw a scatter plot
	# Df0.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2020");
	# Df1.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2019");
	# Df2.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2018");
	# Df3.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2017");
	# Df4.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2016");
	# Df5.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2015");

	# Plots the scatterplots on one page
	fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True, sharex=True, sharey=True)
	fig.suptitle('Average Home Price vs Number of Airbnb per Zipcode by Year');

	Df0.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2020", ax=axes[0,0], color="#B61A53");
	Df1.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2019", ax=axes[0,1], color="#B61A53");
	Df2.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2018", ax=axes[0,2], color="#B61A53");
	Df3.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2017", ax=axes[1,0], color="#B61A53");
	Df4.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2016", ax=axes[1,1], color="#B61A53");
	Df5.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2015", ax=axes[1,2], color="#B61A53");

	plt.show(block=True);

zipcode_scatterplots()


def city_scatterplots():
	Conn = sq.connect(path_to_db)
	#********************************** PER CITY ***************************************************
	query_2020 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2020%' GROUP BY city) SELECT a.ct, a.city, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	query_2019 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2019%' GROUP BY city) SELECT a.ct, a.city, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	query_2018 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2018%' GROUP BY city) SELECT a.ct, a.city, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	query_2017 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2017%' GROUP BY city) SELECT a.ct, a.city, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	query_2016 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2016%' GROUP BY city) SELECT a.ct, a.city, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	query_2015 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2015%' GROUP BY city) SELECT a.ct, a.city, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)

	# Load data into pandas DataFrame
	Df0 = pd.DataFrame(query_2020, columns=['ct', '2020', 'city']);
	Df0.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df0.rename(columns = {'2020':'Avg Home Price'}, inplace = True)

	Df1 = pd.DataFrame(query_2019, columns=['ct', '2019', 'city']);
	Df1.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df1.rename(columns = {'2019':'Avg Home Price'}, inplace = True)

	Df2 = pd.DataFrame(query_2018, columns=['ct', '2018', 'city']);
	Df2.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df2.rename(columns = {'2018':'Avg Home Price'}, inplace = True)

	Df3 = pd.DataFrame(query_2017, columns=['ct', '2017', 'city']);
	Df3.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df3.rename(columns = {'2017':'Avg Home Price'}, inplace = True)

	Df4 = pd.DataFrame(query_2016, columns=['ct', '2016', 'city']);
	Df4.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df4.rename(columns = {'2016':'Avg Home Price'}, inplace = True)

	Df5 = pd.DataFrame(query_2015, columns=['ct', '2015', 'city']);
	Df5.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df5.rename(columns = {'2015':'Avg Home Price'}, inplace = True)

	# Calculate the correlation coefficient and p-value using Pearson's coefficient by YEAR
	X0 = Df0['Avg Home Price'].values
	Y0 = Df0['Number of Airbnbs'].values
	print("2020: " + str(stats.pearsonr(X0, Y0)))

	X1 = Df1['Avg Home Price'].values
	Y1 = Df1['Number of Airbnbs'].values
	print("2019: " + str(stats.pearsonr(X1, Y1)))

	X2 = Df2['Avg Home Price'].values
	Y2 = Df2['Number of Airbnbs'].values
	print("2018: " + str(stats.pearsonr(X2, Y2)))

	X3 = Df3['Avg Home Price'].values
	Y3 = Df3['Number of Airbnbs'].values

	X4 = Df4['Avg Home Price'].values
	Y4 = Df4['Number of Airbnbs'].values
	print("2016: " + str(stats.pearsonr(X4, Y4)))

	X5 = Df5['Avg Home Price'].values
	Y5 = Df5['Number of Airbnbs'].values
	print("2015: " + str(stats.pearsonr(X5, Y5)))

	'''
	X0 = Df0['Avg Home Price']
	Y0 = Df0['Number of Airbnbs']
	model0 = sm.OLS(Y0, X0).fit()
	predictions = model.predict(X0) # make the predictions by the model
	# Print out the statistics
	print(model.summary())
	'''

	# Plots the scatterplots on one page
	fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

	#Df0.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', c='city', colormap='viridis', title= "2020", ax=axes[0,0]);
	Df0.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2020", ax=axes[0,0]);
	Df1.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2019", ax=axes[0,1]);
	Df2.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2018", ax=axes[0,2]);
	Df3.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2017", ax=axes[1,0]);
	Df4.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2016", ax=axes[1,1]);
	Df5.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2015", ax=axes[1,2]);

	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.show(block=True);

# city_scatterplots()

def zipcode_bypricerange_scatterplots():
	# only 200 zipcodes, by diff price range
	Conn = sq.connect(path_to_db)
	#********************************** PER ZIPCODE ***************************************************
	# 200 zipcodes, with highest avg price per person
	query_2020 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2020%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	query_2019 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2019%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	query_2018 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2018%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	query_2017 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2017%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	query_2016 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2016%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	query_2015 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2015%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)

	# 200 zipcodes, with lowest avg price per person
	# query_2020 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2020%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2019 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2019%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2018 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2018%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2017 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2017%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2016 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2016%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2015 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, sum(price)/sum(accommodates) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2015%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)

	# 200 zipcodes, with highest avg price
	# query_2020 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2020%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2019 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2019%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2018 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2018%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2017 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2017%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2016 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2016%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2015 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2015%' GROUP BY zipcode) ORDER by avg_price DESC limit 200) SELECT a.ct, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)

	# 200 zipcodes, with lowest avg price
	# query_2020 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2020%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2019 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2019%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2018 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2018%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2017 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2017%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2016 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2016%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
	# query_2015 = pd.read_sql_query('''WITH a AS (SELECT * from (SELECT COUNT(*) AS ct, avg(price) AS avg_price, zipcode FROM (select * from airbnb where accommodates is not null and accommodates > 0) WHERE last_scraped like '2015%' GROUP BY zipcode) ORDER by avg_price ASC limit 200) SELECT a.ct, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)



	# Load data into pandas DataFrame
	Df0 = pd.DataFrame(query_2020, columns=['ct', '2020']);
	Df0.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df0.rename(columns = {'2020':'Avg Home Price'}, inplace = True)


	Df1 = pd.DataFrame(query_2019, columns=['ct', '2019']);
	Df1.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df1.rename(columns = {'2019':'Avg Home Price'}, inplace = True)

	Df2 = pd.DataFrame(query_2018, columns=['ct', '2018']);
	Df2.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df2.rename(columns = {'2018':'Avg Home Price'}, inplace = True)

	Df3 = pd.DataFrame(query_2017, columns=['ct', '2017']);
	Df3.dropna(subset = ["2017"], inplace=True)
	Df3.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df3.rename(columns = {'2017':'Avg Home Price'}, inplace = True)
	#print(Df3)

	Df4 = pd.DataFrame(query_2016, columns=['ct', '2016']);
	Df4.dropna(subset = ["2016"], inplace=True)
	Df4.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df4.rename(columns = {'2016':'Avg Home Price'}, inplace = True)
	#print(Df4)

	Df5 = pd.DataFrame(query_2015, columns=['ct', '2015']);
	Df5.dropna(subset = ["2015"], inplace=True)
	Df5.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df5.rename(columns = {'2015':'Avg Home Price'}, inplace = True)

	# Removes outliers (defined as points that are more than 3 standard deviations away)
	Df0 = Df0[(np.abs(stats.zscore(Df0)) < 3).all(axis=1)]
	Df1 = Df1[(np.abs(stats.zscore(Df1)) < 3).all(axis=1)]
	Df2 = Df2[(np.abs(stats.zscore(Df2)) < 3).all(axis=1)]
	Df3 = Df3[(np.abs(stats.zscore(Df3)) < 3).all(axis=1)]
	Df4 = Df4[(np.abs(stats.zscore(Df4)) < 3).all(axis=1)]
	Df5 = Df5[(np.abs(stats.zscore(Df5)) < 3).all(axis=1)]

	# Calculate the correlation coefficient and p-value using Pearson's coefficient by YEAR
	X0 = Df0['Avg Home Price'].values
	Y0 = Df0['Number of Airbnbs'].values
	print("2020: " + str(stats.pearsonr(X0, Y0)))

	X1 = Df1['Avg Home Price'].values
	Y1 = Df1['Number of Airbnbs'].values
	print("2019: " + str(stats.pearsonr(X1, Y1)))

	X2 = Df2['Avg Home Price'].values
	Y2 = Df2['Number of Airbnbs'].values
	print("2018: " + str(stats.pearsonr(X2, Y2)))

	X3 = Df3['Avg Home Price'].values
	Y3 = Df3['Number of Airbnbs'].values
	print("2017: " + str(stats.pearsonr(X3, Y3)))

	X4 = Df4['Avg Home Price'].values
	Y4 = Df4['Number of Airbnbs'].values
	print("2016: " + str(stats.pearsonr(X4, Y4)))

	X5 = Df5['Avg Home Price'].values
	Y5 = Df5['Number of Airbnbs'].values
	print("2015: " + str(stats.pearsonr(X5, Y5)))

	# Plots the scatterplots on one page
	fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

	Df0.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2020", ax=axes[0,0]);
	Df1.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2019", ax=axes[0,1]);
	Df2.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2018", ax=axes[0,2]);
	Df3.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2017", ax=axes[1,0]);
	Df4.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2016", ax=axes[1,1]);
	Df5.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2015", ax=axes[1,2]);

	plt.show(block=True);

# zipcode_bypricerange_scatterplots()