######################################################################## Imports
from numpy import hstack
from sklearn.preprocessing import MinMaxScaler
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, BatchNormalization, LeakyReLU , Flatten
from matplotlib import pyplot
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer
import torch
from torch import nn
import math
import matplotlib.pyplot as plt
################################################################ Define Coloumns
COLUMNS_NAMES = {"orderbook": ["sell", "vsell", "buy", "vbuy"],
                 "message": ["time", "event_type", "order_id", "size", "price", "direction"]}
################################################################ Read Apple data
messages = pd.read_csv("../Data/AAPL_2012-06-21_34200000_57600000_message_1.csv", names=COLUMNS_NAMES["message"])
orderbook = pd.read_csv("../Data/AAPL_2012-06-21_34200000_57600000_orderbook_1.csv", names=COLUMNS_NAMES["orderbook"])
#Merge Data
df_combined = messages.copy()
df_combined[COLUMNS_NAMES["orderbook"]] = orderbook
#Only select type 5 and 4 of operations
df_combined = df_combined[(df_combined["event_type"].isin([4,5]))]
#Drop Not Useful Coloumns
Final_df=df_combined.copy()
Final_df.drop(['order_id', 'event_type', 'sell', 'vsell', 'buy', 'vbuy','time',], axis=1, inplace=True)
############################################################ Read Microsoft data
messages1 = pd.read_csv("../Data/MSFT_2012-06-21_34200000_57600000_message_1.csv", names=COLUMNS_NAMES["message"])
orderbook1 = pd.read_csv("../Data/MSFT_2012-06-21_34200000_57600000_orderbook_1.csv", names=COLUMNS_NAMES["orderbook"])
#Merge Data
df_combined1 = messages1.copy()
df_combined1[COLUMNS_NAMES["orderbook"]] = orderbook1
#Only select type 5 and 4 of operations
df_combined1 = df_combined1[(df_combined1["event_type"].isin([4,5]))]
#Drop Not Useful Coloumns
Final_df1=df_combined1.copy()
Final_df1.drop(['order_id', 'event_type', 'sell', 'vsell', 'buy', 'vbuy','time',], axis=1, inplace=True)
############################################################### Read Amazon data
messages2 = pd.read_csv("../Data/AMZN_2012-06-21_34200000_57600000_message_1.csv", names=COLUMNS_NAMES["message"])
orderbook2 = pd.read_csv("../Data/AMZN_2012-06-21_34200000_57600000_orderbook_1.csv", names=COLUMNS_NAMES["orderbook"])
#Merge Data
df_combined2 = messages2.copy()
df_combined2[COLUMNS_NAMES["orderbook"]] = orderbook2
#Only select type 5 and 4 of operations
df_combined2 = df_combined2[(df_combined2["event_type"].isin([4,5]))]
#Drop Not Useful Coloumns
Final_df2=df_combined2.copy()
Final_df2.drop(['order_id', 'event_type', 'sell', 'vsell', 'buy', 'vbuy','time',], axis=1, inplace=True)
################################################################# Read INTC data
messages3 = pd.read_csv("../Data/INTC_2012-06-21_34200000_57600000_message_1.csv", names=COLUMNS_NAMES["message"])
orderbook3 = pd.read_csv("../Data/INTC_2012-06-21_34200000_57600000_orderbook_1.csv", names=COLUMNS_NAMES["orderbook"])
#Merge Data
df_combined3 = messages3.copy()
df_combined3[COLUMNS_NAMES["orderbook"]] = orderbook3
#Only select type 5 and 4 of operations
df_combined3 = df_combined3[(df_combined3["event_type"].isin([4,5]))]
#Drop Not Useful Coloumns
Final_df3=df_combined3.copy()
Final_df3.drop(['order_id', 'event_type', 'sell', 'vsell', 'buy', 'vbuy','time',], axis=1, inplace=True)
############################################################### Read Google data
messages4 = pd.read_csv("../Data/GOOG_2012-06-21_34200000_57600000_message_1.csv", names=COLUMNS_NAMES["message"])
orderbook4 = pd.read_csv("../Data/GOOG_2012-06-21_34200000_57600000_orderbook_1.csv", names=COLUMNS_NAMES["orderbook"])
#Merge Data
df_combined4 = messages4.copy()
df_combined4[COLUMNS_NAMES["orderbook"]] = orderbook4
#Only select type 5 and 4 of operations
df_combined4 = df_combined4[(df_combined4["event_type"].isin([4,5]))]
#Drop Not Useful Coloumns
Final_df4=df_combined4.copy()
Final_df4.drop(['order_id', 'event_type', 'sell', 'vsell', 'buy', 'vbuy','time',], axis=1, inplace=True)
#################################################### Concatinate all data frames
frames = [Final_df, Final_df1, Final_df2, Final_df3, Final_df4]
result = pd.concat(frames)
############################################################### Shuffle the data
result=result.sample(frac=1)
############################################################### Save the dataframe as csv file