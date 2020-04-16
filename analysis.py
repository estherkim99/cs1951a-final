import pandas as pd
import sqlite3 as sq
import numpy as np

path = "./housing.db"
conn = sq.connect(path)
cur = conn.cursor()

tables = ["zillow_zhvi", "zillow_zri"]

for table in tables:
    # table = "zillow_zhvi" # SELECT TABLE

    all_year_field = ""
    for year in range(2015, 2021):
        cur.execute('select * from '+ table +' limit 1')
        mth_num = 0.0
        year_field = ""
        for d in cur.description:
            if d[0][:4] == str(year):
                year_field += "\"" + d[0] + "\" + "
                mth_num += 1.0
        assert(mth_num > 0)
        year_field = "ROUND((" + year_field[:-3] + ")/ " + str(mth_num) + ", 2) as \"" + str(year) + "\""
        all_year_field += year_field +", "
    all_year_field = all_year_field[:-2]

    query_string = "select zipcode, city, state, " + all_year_field + " FROM " + table
    query_df = pd.read_sql_query(query_string, conn)

    # preview 10
    num = str(10)
    query_string_p = "select zipcode, city, state, " + all_year_field + " FROM " + table + " limit " + num
    query_df_p = pd.read_sql_query(query_string_p, conn)
    print(query_df)

    query_df.to_sql(table + "_yearavg", con=conn, if_exists="replace")

    conn.commit()

conn.close()