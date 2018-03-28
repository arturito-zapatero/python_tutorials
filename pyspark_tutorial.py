from pyspark.sql import functions as F
from pyspark.sql.window import Window
import datetime as dt
import numpy as np
import pandas as pd
import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext
from pyspark.sql import SparkSession

os.environ["HADOOP_HOME"] = "/usr/lib/spark"
print os.environ["HADOOP_HOME"]

#configure spark settings
conf = (SparkConf()
        .setMaster("yarn-client")
        .setAppName("Vueling Test0")
#        .set("spark.executor.instances", 4)
#        .set("spark.executor.memory", "8g")
        .set("spark.driver.memory", "8g")
        .set("spark.ui.enabled", "false")
        )

#connect to spark context
sc = SparkContext(conf=conf)

#other way: works in Vueling:)
#from pyspark.sql import SparkSession
#spark = SparkSession.builder.master("yarn-client").appName("spark-app")


spark = SparkSession.builder.appName("booking_aggregator").\
enableHiveSupport().getOrCreate()

spark_query = "SELECT oandd, substr(departuredate,1,10) as departuredate, flightnumber, pnr, pax_nbr, collect_set(serviceclass) as serviceclass, collect_set(fareclass) as fareclass, collect_set(family_fare) as familyfare, SUM(pax) as pax, SUM(cdiscount) as cdiscount, SUM(tax_2) as tax2 from (SELECT * from (SELECT *, RANK() OVER (PARTITION BY pnr, oandd, sales_date, pax_nbr ORDER BY importdatetime DESC) as rk FROM ancillary.ds_vb_bookings_rolling {}) q1 WHERE rk=1) q2 GROUP BY oandd, SUBSTR(departuredate,1,10), flightnumber, pnr, pax_nbr".format(condition_filter) 


df = spark.sql(spark_query)



# Create a sql context with hive
sqlContext = HiveContext(sc)

sqlContext.sql("use pv_konfigurator")
#show all hive tables in current folder
sqlContext.sql("show tables").collect()

#create spark d.f. from hive table
baseline_results = sqlContext.table("pv_konfigurator.de_pv_nan_baseline")
baseline_results.show()

#dont know what it does
baseline_results_local1 = baseline_results.collect

#create local pandas d.f. from spark d.f.
baseline_results_local = baseline_results.toPandas()
baseline_results_local.shape
baseline_results_local.head()

#saves pandas df back as spark df
new_spark = sqlContext.createDataFrame(baseline_results_local)
new_spark.show()
new_spark.cache()
new_spark.columns
new_spark.head(3)

#save the table in HDFS
baseline_results.write.save("hdfs://LIDL/lidl/sandboxb_team1/baseline_test/test.orc")
#df.write.save('/target/path/', format='parquet', mode='append')
#baseline_results.write("hdfs://LIDL/lidl/pv_konfigurator/....orc")
#in command line: hdfs dfs -ls hdfs://LIDL/lidl/pv_konfigurator

#create a table in Hive
sqlContext.sql("create table sandboxb_team1.test_c (client_id smallint,\
 nan int, dat_formdate date, abverkauf double, umsatz double, preis double, anzahl_filialen int, uwg int, marke smallint, mhd_tage int, artikelfamilie string)")

#create hive table from file saved in hdfs, not tested
sqlContext.sql("create table sandboxb_team1.test_c (client_id smallint,\
 nan int, dat_formdate date, abverkauf double, umsatz double, preis double, anzahl_filialen int, uwg int, marke smallint, mhd_tage int, artikelfamilie string)\
  load data from hdfs://LIDL/lidl/sandboxb_team1/baseline_test")
#not tested
sqlContext.sql("LOAD DATA LOCAL INPATH 'examples/src/main/resources/kv1.txt' INTO TABLE src")


#saves file from spark d.f. into hive table
baseline_results.write.format("orc").saveAsTable("sandboxb_team1.test_b")

#get data from ext file, not tested
#csv_data = sc.textFile("file:///home/username/names.csv")