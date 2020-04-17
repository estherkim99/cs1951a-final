import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3 as sq
from scipy import stats
import statsmodels.api as sm

path_to_db = "./data/housing.db"
Conn = sq.connect(path_to_db)

#
def cities_by_year():
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
	#********************************** PER CITY ***************************************************
	# query_2020 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2020%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2020") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	# query_2019 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2019%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2019") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	# query_2018 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2018%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2018") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	# query_2017 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2017%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2017") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	# query_2016 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2016%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2016") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	# query_2015 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2015%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2015") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)

	# WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2020%' GROUP BY cityname) SELECT a.ct, avg, b.city FROM ((select avg("2020") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;

	# print(query_2020.describe())
	# exit()
	# query_2019 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2019%' GROUP BY cityname) SELECT a.ct, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.cityname) GROUP BY a.cityname;''', Conn)
	# query_2018 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2018%' GROUP BY city) SELECT a.ct, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	# query_2017 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2017%' GROUP BY city) SELECT a.ct, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	# query_2016 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2016%' GROUP BY city) SELECT a.ct, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)
	# query_2015 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2015%' GROUP BY city) SELECT a.ct, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;''', Conn)



	#********************************** PER ZIPCODE ***************************************************
	all_cities = ["Boston", "Chicago", "San Francisco", "New York", "Nashville", "Los Angeles", "Austin", "Seattle", "Denver", "Asheville"]
	# all_cities = ["New York"]
	for city in all_cities:
		print("calculating correlation for... ",city)

		query20 = "WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '{}%' and cityname = '{}' GROUP BY zipcode) SELECT a.ct, \"{}\" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;".format(2020, city, 2020)
		query19 = "WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '{}%' and cityname = '{}' GROUP BY zipcode) SELECT a.ct, \"{}\" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;".format(2019, city, 2019)
		query18 = "WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '{}%' and cityname = '{}' GROUP BY zipcode) SELECT a.ct, \"{}\" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;".format(2018, city, 2018)
		query17 = "WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '{}%' and cityname = '{}' GROUP BY zipcode) SELECT a.ct, \"{}\" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;".format(2017, city, 2017)
		query16 = "WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '{}%' and cityname = '{}' GROUP BY zipcode) SELECT a.ct, \"{}\" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;".format(2016, city, 2016)
		query15 = "WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '{}%' and cityname = '{}' GROUP BY zipcode) SELECT a.ct, \"{}\" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;".format(2015, city, 2015)
		# print(query20)
		query_2020 = pd.read_sql_query(query20, Conn)
		query_2019 = pd.read_sql_query(query19, Conn)
		query_2018 = pd.read_sql_query(query18, Conn)
		query_2017 = pd.read_sql_query(query17, Conn)
		query_2016 = pd.read_sql_query(query16, Conn)
		query_2015 = pd.read_sql_query(query15, Conn)
		# query_2020 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2020%' and cityname = 'New York' GROUP BY zipcode) SELECT a.ct, "2020" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2019 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2019%' and cityname = 'New York' GROUP BY zipcode) SELECT a.ct, "2019" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2018 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2018%' and cityname = 'New York' GROUP BY zipcode) SELECT a.ct, "2018" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2017 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2017%' and cityname = 'New York' GROUP BY zipcode) SELECT a.ct, "2017" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2016 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2016%' and cityname = 'New York' GROUP BY zipcode) SELECT a.ct, "2016" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2015 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2015%' and cityname = 'New York' GROUP BY zipcode) SELECT a.ct, "2015" as avg FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)

		# query_2020 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2020%' GROUP BY zipcode) SELECT a.ct, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2019 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2019%' GROUP BY zipcode) SELECT a.ct, "2019" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2018 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2018%' GROUP BY zipcode) SELECT a.ct, "2018" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2017 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2017%' GROUP BY zipcode) SELECT a.ct, "2017" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2016 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2016%' GROUP BY zipcode) SELECT a.ct, "2016" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)
		# query_2015 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, zipcode FROM airbnb WHERE last_scraped like '2015%' GROUP BY zipcode) SELECT a.ct, "2015" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.zipcode=a.zipcode) GROUP BY a.zipcode;''', Conn)


		# Load data into pandas DataFrame
		Df0 = pd.DataFrame(query_2020, columns=['ct', 'avg']);
		Df0.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
		Df0.rename(columns = {'avg':'Avg Home Price'}, inplace = True)


		Df1 = pd.DataFrame(query_2019, columns=['ct', 'avg']);
		Df1.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
		Df1.rename(columns = {'avg':'Avg Home Price'}, inplace = True)

		Df2 = pd.DataFrame(query_2018, columns=['ct', 'avg']);
		Df2.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
		Df2.rename(columns = {'avg':'Avg Home Price'}, inplace = True)

		Df3 = pd.DataFrame(query_2017, columns=['ct', 'avg']);
		Df3.dropna(subset = ["avg"], inplace=True)
		Df3.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
		Df3.rename(columns = {'avg':'Avg Home Price'}, inplace = True)
		#print(Df3)

		Df4 = pd.DataFrame(query_2016, columns=['ct', 'avg']);
		Df4.dropna(subset = ["avg"], inplace=True)
		Df4.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
		Df4.rename(columns = {'avg':'Avg Home Price'}, inplace = True)
		#print(Df4)

		Df5 = pd.DataFrame(query_2015, columns=['ct', 'avg']);
		Df5.dropna(subset = ["avg"], inplace=True)
		Df5.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
		Df5.rename(columns = {'avg':'Avg Home Price'}, inplace = True)

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
		# fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
		fig.suptitle('Average Home Price vs Number of Airbnb of each City, by Year for city ='+ city);

		Df0.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2020", ax=axes[0,0]);
		Df1.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2019", ax=axes[0,1]);
		Df2.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2018", ax=axes[0,2]);
		Df3.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2017", ax=axes[1,0]);
		Df4.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2016", ax=axes[1,1]);
		Df5.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2015", ax=axes[1,2]);

		plt.show(block=True);

# zipcode_scatterplots()


def city_scatterplots():
	# WITH a AS (SELECT COUNT(*) AS ct, city FROM airbnb WHERE last_scraped like '2020%' GROUP BY city) SELECT a.ct, a.city, "2020" FROM (zillow_zhvi_yearavg JOIN a ON zillow_zhvi_yearavg.city=a.city) GROUP BY a.city;
	#********************************** PER CITY ***************************************************
	query_2020 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2020%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2020") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	query_2019 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2019%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2019") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	query_2018 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2018%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2018") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	query_2017 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2017%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2017") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	query_2016 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2016%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2016") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)
	query_2015 = pd.read_sql_query('''WITH a AS (SELECT COUNT(*) AS ct, cityname FROM airbnb WHERE last_scraped like '2015%' GROUP BY cityname) SELECT a.ct, avg FROM ((select avg("2015") as "avg", city from zillow_zhvi_yearavg group by city) as b JOIN a ON b.city=a.cityname) GROUP BY a.cityname;''', Conn)

	# Load data into pandas DataFrame
	Df0 = pd.DataFrame(query_2020, columns=['ct', 'avg']);
	Df0.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df0.rename(columns = {'avg':'Avg Home Price'}, inplace = True)
	# Df0 = Df0[(np.abs(stats.zscore(Df0)) < 3).all(axis=1)]
	#print(Df0);

	Df1 = pd.DataFrame(query_2019, columns=['ct', 'avg']);
	Df1.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df1.rename(columns = {'avg':'Avg Home Price'}, inplace = True)
	#print(Df1);

	Df2 = pd.DataFrame(query_2018, columns=['ct', 'avg']);
	Df2.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df2.rename(columns = {'avg':'Avg Home Price'}, inplace = True)
	#print(Df2);

	Df3 = pd.DataFrame(query_2017, columns=['ct', 'avg']);
	Df3.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df3.rename(columns = {'avg':'Avg Home Price'}, inplace = True)
	#print(Df3);

	Df4 = pd.DataFrame(query_2016, columns=['ct', 'avg']);
	Df4.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df4.rename(columns = {'avg':'Avg Home Price'}, inplace = True)
	#print(Df4);

	Df5 = pd.DataFrame(query_2015, columns=['ct', 'avg']);
	Df5.rename(columns = {'ct':'Number of Airbnbs'}, inplace = True)
	Df5.rename(columns = {'avg':'Avg Home Price'}, inplace = True)
	#print(Df5);

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

	# Plots the scatterplots on one page
	fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True, sharex=True, sharey=True)
	# fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
	fig.suptitle('Average Home Price vs Number of Airbnb of each City, by Year');

	#Df0.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', c='city', colormap='viridis', title= "2020", ax=axes[0,0]);
	Df0.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2020", ax=axes[0,0]);
	Df1.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2019", ax=axes[0,1]);
	Df2.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2018", ax=axes[0,2]);
	Df3.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2017", ax=axes[1,0]);
	Df4.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2016", ax=axes[1,1]);
	Df5.plot.scatter(x='Number of Airbnbs', y='Avg Home Price', title= "2015", ax=axes[1,2]);

	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

	plt.show(block=True);

city_scatterplots()

def zipcode_bypricerange_scatterplots():
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

Conn.close()