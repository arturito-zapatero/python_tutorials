# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:18:30 2018

@author: aszewczyk
"""
import pandas as pd
import pyodbc

user = 'vuelingbcn\\artur.szewczyk' 
pwd  = 'Pimpek23$'

server = "10.218.7.178;instance=ODS"
database = "Vueling_CALCODS"
user = "l.s3extract"
pwd = 'cc#5oy5NgtEox6wYg4#xSt'


conn = pyodbc.connect("DRIVER={}; "
"SERVER={}; "
"DATABASE={}; "
"UID={}; "
"PWD={}".format('{ODBC Driver 11 for SQL Server}', 
     server, 
     database, 
     user, 
     pwd))


sql = 'SELECT Airport, country, is_eu, zone FROM dbo.DimAirport'

sql = 'SELECT * FROM information_schema.columns ' \
'WHERE table_schema = \'{}\' AND table_name = \'{}\';'.format(schema, table)

data = pd.read_sql(sql, conn)
