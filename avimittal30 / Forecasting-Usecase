# Databricks notebook source
# MAGIC %md
# MAGIC <h1>Table of Contents<span class="tocSkip"></span></h1>
# MAGIC <div class="toc"><ul class="toc-item"><li><span><a href="#Demand-Forecasting-Model-in-Azure-Data-Bricks" data-toc-modified-id="Demand-Forecasting-Model-in-Azure-Data-Bricks-1">Demand Forecasting Model in Azure Data Bricks</a></span></li></ul></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demand Forecasting Model in Azure Data Bricks 
# MAGIC ###### Deepanker Anand, Aviral Mittal
# MAGIC ###### Jan 2021
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# MAGIC %md
# MAGIC Import Python Libraries

# COMMAND ----------



 !pip install plotly
 !pip install --upgrade plotly
 !pip install fbprophet

import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random
import seaborn as sns
from fbprophet import Prophet
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

!pip install plotly



# COMMAND ----------

!pip install cufflinks --upgrade
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

# COMMAND ----------

import os
print(os.getcwd())
print(os.path.abspath('../tables'))

# Change working directory
os.chdir('/dbfs/FileStore/shared_uploads/deepanker.anand@absolutdata.com/')
print(os.getcwd())

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/train_demand_forecasting.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "train_demand_forecasting_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `train_demand_forecasting_csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.

# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.

# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "train_demand_forecasting_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# Make connection to the Azure Blob Storage

storage_account_name = "deepankerstorage1"
storage_account_access_key = "GScZye6mFi/5Jx884+6GTpOOhy16ka8jtj/Y5not/ZKmhUyq6CJQFSMHc05i/uFgRhzJik7cvlpy8Q5eToyfhQ=="
blobContainerName="datasets"

# COMMAND ----------

file_location = "wasbs://deepankerstorage1.blob.core.windows.net/mycontainer/demand_forecasting/train_demand_forecasting.csv".format(blobContainerName, storage_account_name)
file_type = "csv"

# https://deepankerstorage1.blob.core.windows.net/train_demand_forecasting.csv

# wasbs://{}@{}.blob.core.windows.net/mycontainer/demand_forecasting/train_demand_forecasting.csv

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key) #spark context 

# COMMAND ----------

### Commenting this code as it throws error (RERUN THIS ONCE THIS ISSUE GETS RESOLVED)

# df = spark.read.format(file_type).option("inferSchema", "true").load(file_location)

# COMMAND ----------

 # load fbprophet library
# dbutils.library.installPyPI('FBProphet', version='0.5') # find latest version of fbprophet here: https://pypi.org/project/fbprophet/
# dbutils.library.installPyPI('holidays','0.9.12') # this line is in response to this issue with fbprophet 0.5: 
# https://github.com/facebook/prophet/issues/1293

# dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.types import *

# structure of the training data set
train_schema = StructType([
  StructField('date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('sales', IntegerType())
  ])

# read the training file into a dataframe
train = spark.read.csv(
  '/FileStore/tables/train_demand_forecasting.csv', 
  header=True, 
  schema=train_schema
  )

# make the dataframe queriable as a temporary view
train.createOrReplaceTempView('train')

# COMMAND ----------

train.show()

# COMMAND ----------

train.printSchema()

# COMMAND ----------

display(train)

# COMMAND ----------

# convert pyspark dataframe to pandas dataframe
train=train.toPandas()
type(train)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Read raw datasets from DBFS location

# COMMAND ----------

# Reading file from DBFS Storage 
import pandas as pd

# df1 = pd.read_csv("/dbfs/FileStore/shared_uploads/deepanker.anand@absolutdata.com/sample_submission.csv")
# df2 = pd.read_csv("/dbfs/FileStore/shared_uploads/deepanker.anand@absolutdata.com/test_demand_forecasting.csv")
#df3 = pd.read_csv("/dbfs/FileStore/shared_uploads/deepanker.anand@absolutdata.com/train_demand_forecasting.csv")
df3 = pd.read_csv("/dbfs/FileStore/shared_uploads/aviral.mittal@absolutdata.com/train-1.csv")

# COMMAND ----------

train=df3
train.head()

# COMMAND ----------

train.shape

# COMMAND ----------

train.info()

# COMMAND ----------

train.columns

# COMMAND ----------

# count no of missing values in entire dataframe 
train.isnull().sum().sum()

# COMMAND ----------

# count no of missing values in each column of dataframe 
train.isnull().sum()

# COMMAND ----------

# Descriptive statistics   
train.describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

# COMMAND ----------

train.dtypes

# COMMAND ----------

# train=train.sort_values(['date'])

# COMMAND ----------

print(len(train)) # Count total cells for each column  
print(train.count()) # Count non-NA cells for each column 

# COMMAND ----------

train['store'].unique().tolist()  

# COMMAND ----------

train['store'].value_counts(normalize=True, sort=True, ascending=False) 

# COMMAND ----------

import pandas as pd

# Converting sales to nummeric column
train["sales"]=train["sales"].astype(int)
train["year"] = pd.to_datetime(train["date"]).dt.year
train["month"] = pd.to_datetime(train["date"]).dt.month
train['weekday'] = pd.to_datetime(train["date"]).dt.dayofweek
train['quarter'] = pd.to_datetime(train["date"]).dt.quarter
train['dayofmonth'] = pd.to_datetime(train["date"]).dt.day
train['dayofyear'] = pd.to_datetime(train["date"]).dt.dayofyear
train["month_year"] = pd.to_datetime(train["year"] * 10000 + train["month"] * 100 + 1, format="%Y%m%d")


train.tail()

# COMMAND ----------

# checking for missing values 
count_nan = len(train) - train.count()
count_nan

train.dtypes

# COMMAND ----------

# drop any missing records
train = train.dropna()

# COMMAND ----------

train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC Perform Exploratory Data Analysis (EDA) on raw dataset

# COMMAND ----------

# MAGIC %md
# MAGIC * Looking at time perspective 
# MAGIC * Analyse from Item perspective
# MAGIC * Analyse from Store perspective

# COMMAND ----------

import matplotlib.pyplot as plt

plt.hist(train["sales"])
plt.show()

# COMMAND ----------

import seaborn as sns

mean_sales = train.groupby(["date"], as_index=False)
mean_sales = mean_sales[["sales"]].sum()
#mean_sales["idx"] = mean_sales.index
g = sns.relplot(x="date", y="sales", data=mean_sales, kind="line")

# COMMAND ----------

# mean sales for each item 
sales_vs_item=train[['item','sales']]
sales_vs_item.set_index('item',inplace=True)
sales_vs_item

data=sales_vs_item.groupby(['item'], as_index=True).mean()
data
sns.distplot(data['sales'])

# COMMAND ----------

#import plotly as py 
# Time Series Analysis for Store 10 and Item 40
# performing time series analysis for particular item of store
print('Before Filtering '+str(train.shape))

# let consider particular item and store
# store=10
# item=40

sample=train[train.store==10]
sample=sample[sample.item==40]
print('After Filtering '+str(sample.shape))

sns.lineplot(x="date", y="sales", data=sample)

# COMMAND ----------

# MAGIC %sql  --annual trend in unit sales
# MAGIC 
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1; 

# COMMAND ----------

# MAGIC %sql  -- If we aggregate the data around the individual months in each year, a distinct yearly seasonal pattern is observed which seems to grow in # scale with overall growth in sales
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1;

# COMMAND ----------

train['weekday'].value_counts()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC --drop table x;
# MAGIC 
# MAGIC --create table x as
# MAGIC SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM train
# MAGIC   GROUP BY 1;
# MAGIC   
# MAGIC --select * from x limit 10;  

# COMMAND ----------

# MAGIC %sql -- Aggregating the data at a weekday level, a pronounced weekly seasonal pattern is observed with a peak on Sunday (weekday 0), a hard drop on Monday (weekday 1) and then a steady pickup over the week heading back to the Sunday high. This pattern seems to be pretty stable across the five years of observations:
# MAGIC 
# MAGIC -- SELECT YEAR(date) as year,
# MAGIC --   CAST(DATE_FORMAT(date, 'u') as Integer) % 7 as weekday,
# MAGIC --   avg(sales) as sales
# MAGIC -- FROM (
# MAGIC --   SELECT 
# MAGIC --     date,
# MAGIC --     SUM(sales) as sales
# MAGIC --   FROM train
# MAGIC --   GROUP BY date
# MAGIC --      ) x
# MAGIC -- GROUP BY 1, CAST(DATE_FORMAT(date, 'u') as Integer)
# MAGIC -- ORDER BY 1, 2;

# COMMAND ----------

sales_vs_item=train[['item','sales']]
sales_vs_item.set_index('item',inplace=True)


plt.figure(figsize=(10,5))
sns.distplot(sales_vs_item['sales'])

# Some items has high demand in comparision to other

# COMMAND ----------

# Avg. sales for each item 
data=sales_vs_item.groupby(['item']).mean()
plt.figure(figsize=(20,8))
sns.barplot(data.index,data['sales'])

# COMMAND ----------

# Sales trend over the months and year

# train['Month']=train['date'].dt.month
# train['Year']=train['date'].dt.year
plt.figure(figsize=(10,30))
sns.catplot(data = train, x ='month', y = "sales",row = "year")
plt.show()

# COMMAND ----------

# Understand sales data distribution across the stores
import numpy as np
store_df = train.copy()
sales_pivoted_df = pd.pivot_table(store_df, index='store', values=['sales'], columns='item', aggfunc=np.mean)
# Pivoted dataframe
display(sales_pivoted_df)

# 10 rows × 50 columns

# COMMAND ----------

store_data=train[['store','sales']]
store_data.set_index('store',inplace=True)
store_data=store_data.groupby('store').mean()
store_data['store'] = store_data.index
store_data.head()

# COMMAND ----------

plt.figure(figsize=(15,5))
store_data.sort_values('sales')
sns.barplot(x=store_data.index,y=store_data['sales'],order=store_data.sort_values('sales').store) # Bars are ordered in ascending order  

# COMMAND ----------

# MAGIC %md
# MAGIC Build a Forecasting Model

# COMMAND ----------

train_data=train.rename(columns={'date':'ds','sales':'y'})
train_data.head(5)

# COMMAND ----------

 !pip install fbprophet
 !pip install --upgrade plotly 

# COMMAND ----------

from fbprophet import Prophet
import plotly.offline as py
import logging

# disable informational messages from fbprophet
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

# Dataframe to be used for all the algorithms
filtered=train_data[(train_data['item']==50) & (train_data['store']==10)]
traindf=filtered[filtered["year"]<2017]
testdf=filtered[filtered['year']==2017]

traindf_ts=traindf[['month_year','y']]
traindf_ts=traindf_ts.groupby("month_year").sum()

testdf_ts=testdf[['month_year','y']]
testdf_ts=testdf_ts.groupby("month_year").sum()



# COMMAND ----------


filtered_train=filtered[filtered['year']<2017] 


filtered_train=filtered_train[['ds','y']]

# set model parameters
m = Prophet(
  interval_width=0.95,
  growth='linear',
  daily_seasonality=False,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode='multiplicative'
  )

# fit the model to historical data
m.fit(filtered_train)

# COMMAND ----------

# Define a dataset including both historical dates & 365-days beyond the last available date
future_pd = m.make_future_dataframe(
  periods=365, 
  freq='d', 
  include_history=True
  )

future_pd.tail()

# COMMAND ----------

# predict over the dataset
forecast_pd = m.predict(future_pd)
forecast_pd[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# COMMAND ----------

# Check the general and seasonal trends in our forecast model 
trends_fig = m.plot_components(forecast_pd)
trends_fig

# COMMAND ----------

predict_fig = m.plot( forecast_pd, xlabel='date', ylabel='sales')
display(predict_fig)

# COMMAND ----------

# MAGIC %md
# MAGIC The black dots represent our actuals with the darker blue line representing our predictions and the lighter blue band representing our (95%) uncertainty interval.

# COMMAND ----------

# Check how our actual and predicted data line up as well as a forecast for the future

predict_fig = m.plot( forecast_pd, xlabel='date', ylabel='sales')

# adjust figure to display dates from last year + the 365 day forecast
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+365.0), xlim[1]-365.0)
predict_fig.axes[0].set_xlim(new_xlim)

display(predict_fig)

# COMMAND ----------

filtered['ds'] = filtered['ds'].astype('datetime64[ns]')

# COMMAND ----------

# Calculate Model Performance Metrics to evaulate the forecast model 
# Mean Absolute Error, Mean Squared Error and Root Mean Squared Error values for the predicted relative to the actual values in our dataset

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date

import pandas as pd


# get historical actuals & predictions for comparison
actuals_pd = filtered_train[ pd.to_datetime(filtered_train['ds']) < pd.to_datetime(date(2017, 1, 1)) ]['y']
predicted_pd = forecast_pd[ pd.to_datetime(forecast_pd['ds']) < pd.to_datetime(date(2017, 1, 1)) ]['yhat']

# calculate evaluation metrics
mae = mean_absolute_error(actuals_pd, predicted_pd)
mse = mean_squared_error(actuals_pd, predicted_pd)
rmse = sqrt(mse)

# print metrics to the screen
print( '\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse) )

# COMMAND ----------

filtered.head() # completed data for item=50 and store=10 from 2013-2017

# COMMAND ----------

filtered_future=filtered[filtered['year']==2017] # filter the forecast period from complete dataset

a=filtered_future[['ds','y']] # actual
b=forecast_pd[['ds','yhat', 'yhat_lower', 'yhat_upper']] # predicted

# set index as ds in both datasets 
filtered.set_index('ds')
b.set_index('ds')

#filtered['ds']=filtered['ds'].astype('datetime64[ns]')
#b['ds']=b['ds'].astype('datetime64[ns]')
#final=filtered.join(b, on="ds")

filtered['ds'] = filtered['ds'].astype('datetime64[ns]')
filtered.set_index('ds')
b.set_index('ds')
final=filtered.merge(b, how='inner')


#a['ds'] = pd.to_datetime(a['ds'],  format='%Y%m%d')
#a['ds'] =  pd.to_datetime(a['ds'])
#a.join(b, on='ds')

#pd.concat([a,b])
#pd.concat(left=a, right=b, how="inner")
#a['ds']=a['ds']

# COMMAND ----------

final.head()

# COMMAND ----------

final.shape

# COMMAND ----------

# Export forecast model results into CSV file
final.to_csv('/dbfs/FileStore/shared_uploads/deepanker.anand@absolutdata.com/fbp_forecast_1.csv')

# COMMAND ----------

# MAPE 
import numpy as np

#final.to_csv('/dbfs/FileStore/fbp_forecast_1.csv')

y_actual=final['y']
y_pred=final['yhat']
MAPE=np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
print(MAPE)

# COMMAND ----------


filtered.head()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

filtered_train_arima=filtered[filtered["year"]<2017]
filtered_test_arima=filtered[filtered["year"]>=2017]

filtered_train_arima.tail()

filtered_ts=filtered_train_arima[["month_year", "y"]]
filtered_ts=filtered_ts.groupby("month_year").sum()
filtered_ts_log=filtered_ts.copy()
filtered_ts_log['y']=np.log(filtered_ts_log['y'])


filtered_test_ts=filtered_test_arima[["month_year", "y"]]
filtered_test_ts=filtered_test_ts.groupby("month_year").sum()

filtered_test_ts

# COMMAND ----------

!pip install pmdarima
#from pyramid.arima import auto_arima
from pmdarima import auto_arima


# COMMAND ----------

#Arima Model Fitting

model1 = auto_arima(filtered_ts, trace=True, error_action='ignore', seasonal=True, suppress_warnings=True)
model1.fit(filtered_ts)

model2 = auto_arima(filtered_ts_log, trace=True, error_action='ignore',seasonal=True, suppress_warnings=True)
model2.fit(filtered_ts_log)


# COMMAND ----------



# COMMAND ----------

# Arima Forecast Generation

#from plotly.offline import plot
#import matplotlib.pyplot as plt

forecast1 = model1.predict(n_periods=12)
#plt.plot(forecast1)
filtered_ts['prediction']=model1.predict_in_sample(filtered_ts)
filtered_test_ts['prediction']=forecast1

future_forecast1 = pd.DataFrame(filtered_test_ts,index = filtered_test_ts.index,columns=["prediction"])
future_forecast1['prediction']=forecast1
fcst_plot=pd.concat([filtered_ts,filtered_test_ts], axis=0)
#sns.lineplot(data=fcst_plot, x="month_year", y="y")
#fcst_plot.plot()
#plt.show()
fcst_plot['month_year']=fcst_plot.index
display(fcst_plot)

# COMMAND ----------

# import matplotlib.pyplot as plt
# ax = plt.gca()

# fcst_plot.plot(kind='line',x='index',y='y',ax=ax)
# df.plot(kind='line',x='index',y='prediction', color='red', ax=ax)

# COMMAND ----------

# Plot with log
forecast2 = np.exp(model2.predict(n_periods=12))
plt.plot(forecast2)
forecast2


# COMMAND ----------

import numpy as np
filtered_test_ts['y_pred']=forecast1
y_actual=filtered_test_ts['y']
y_pred=filtered_test_ts['y_pred']
MAPE=np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
print("MAPE without log:", MAPE)

smape1=pm.metrics.smape(filtered_test_ts['y'],forecast1)
print("SMAPE without log:",smape1)



filtered_test_ts['y_pred']=forecast2
y_actual=filtered_test_ts['y']
#y_pred=filtered_test_ts['y_pred']
MAPE2=np.mean(np.abs((y_actual - forecast2) / y_actual)) * 100
print("MAPE with log:",MAPE2)

smape2=pm.metrics.smape(filtered_test_ts['y'],forecast2)
print("SMAPE with log:", smape2)



# COMMAND ----------

model.summary()

# COMMAND ----------

# Holtwinter Implementation

from statsmodels.tsa.holtwinters import ExponentialSmoothing
#import pmdarima as pm
import numpy as np

traindf.head()
traindf_ts=traindf[['month_year','y']]
traindf_ts=traindf_ts.groupby("month_year").sum()

holt_model = ExponentialSmoothing(traindf_ts, trend="add",seasonal="add", seasonal_periods=12)
holt_fit=holt_model.fit()
pred=holt_fit.forecast(12)

testdf_ts=testdf[['month_year','y']]
testdf_ts=testdf_ts.groupby("month_year").sum()
testdf_ts['pred']=pred
smape1=pm.metrics.smape(testdf_ts['y'],pred)



traindf_ts_log=traindf_ts
traindf_ts_log['y']=np.log(traindf_ts['y'])
holt_model = ExponentialSmoothing(traindf_ts, trend="add",seasonal="add", seasonal_periods=12)
holt_fit=holt_model.fit()
pred2=np.exp(holt_fit.forecast(12))
smape2=pm.metrics.smape(testdf_ts['y'],pred2)


print("SMAPE without log:",smape1)
print("SMAPE with log:",smape2)



# COMMAND ----------

!pip install tbats

# COMMAND ----------

traindf.head()

# COMMAND ----------

# Tbats Implementation

from tbats import TBATS

estimator1 = TBATS(seasonal_periods=[12,24])
model1 = estimator1.fit(traindf_ts)
forecast1=model1.forecast(12)

#model.predict_in_sample(traindf_ts)
testdf_ts['pred1'] = forecast1


estimator2 = TBATS()
model2 = estimator2.fit(traindf_ts)
forecast2=model2.forecast(12)

#model.predict_in_sample(traindf_ts)
#testdf_ts['pred2'] = forecast2

mape1=np.mean(np.abs((testdf_ts['y'] - forecast1) / testdf_ts['y'])) * 100
mape2=np.mean(np.abs((testdf_ts['y'] - forecast2) / testdf_ts['y'])) * 100

tbats_train_df=traindf_ts.copy()
tbats_train_df['pred1']=np.nan

tbats_test_df=testdf_ts.copy()
tbats_test_df['pred1']=forecast1

tbatsdf=pd.concat([tbats_train_df, tbats_test_df], axis=0)


tbatsdf['month_year']=filtered['month_year'].unique()
display(tbatsdf)

#print("MAPE1", mape1)
#print("MAPE2", mape2)

# COMMAND ----------

#Tbats Evaluation

print("MAPE1 with Seasonal Periods", mape1)
print("MAPE2 with default ", mape2)

# COMMAND ----------

!pip install xgboost

# COMMAND ----------



# COMMAND ----------

#XGBoost Implementation

import xgboost as xgb
import numpy as np
xgbdf_train=filtered[filtered['year']<2017]
xgbdf_test=filtered[filtered['year']>=2017]

x_train=xgbdf_train[['month', 'weekday', 'quarter', 'year', 'dayofmonth', 'dayofyear']]
y_train=xgbdf_train['y']

x_test=xgbdf_test[['month', 'weekday', 'quarter', 'year',  'dayofmonth', 'dayofyear']]
y_test=xgbdf_test['y']

reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds=5, #stop if 50 consequent rounds without decrease of error
        verbose=False) # Change verbose to True if you want to see it train


y_test_pred = reg.predict(x_test)
y_train_pred= reg.predict(x_train)
y_train.index=filtered[filtered['year']<2017].ds

df_xgb = pd.DataFrame({'date' : []})
df_xgb['date']=filtered.ds
df_xgb['y']=filtered['y']
df_xgb['y_pred']=np.concatenate([y_train_pred, y_test_pred])
df_xgb['month_year']=filtered['month_year']
display(df_xgb)

# COMMAND ----------

# XGBoost Evaluation

df_xgb_monthly=df_xgb.groupby("month_year", as_index=False).sum()
display(df_xgb_monthly)
print("MAPE:", np.mean(np.abs((xgbdf_test['y'] - y_test_pred) / xgbdf_test['y'])) * 100)


# COMMAND ----------

!pip install keras
!pip install tensorflow

# COMMAND ----------

# libraries for LSTM implementation
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense, LSTM



# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler


lstmdf_train=filtered[filtered['year']<2017]
lstmdf_test=filtered[filtered['year']>=2017]

scaler=MinMaxScaler()

lstmdf_train=lstmdf_train[['ds', 'y']]
lstmdf_train.set_index(['ds'], inplace=True)
lstmdf_test=lstmdf_test[['ds', 'y']]
lstmdf_test.set_index(['ds'], inplace=True)

scaler.fit(lstmdf_train)

scaled_train=scaler.transform(lstmdf_train)
scaled_test=scaler.transform(lstmdf_test)

train_data_gen = TimeseriesGenerator(scaled_train,scaled_train,length=120, sampling_rate=1,stride=1,batch_size=3)
train_data_gen


# COMMAND ----------

