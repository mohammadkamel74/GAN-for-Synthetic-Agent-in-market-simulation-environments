from agent.examples.SubscriptionAgent import SubscriptionAgent
import pandas as pd
from copy import deepcopy

from keras.layers import Dense, Conv1D, Dropout, BatchNormalization, LeakyReLU , Flatten ,LSTM
import keras
from random import randrange


#from keras.models import load_model
from tensorflow.keras.models import load_model
from numpy import hstack
from sklearn.preprocessing import MinMaxScaler
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, BatchNormalization, LeakyReLU , Flatten ,LSTM
from matplotlib import pyplot
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer
#import torch
#from torch import nn
import math
import matplotlib.pyplot as plt
import joblib
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(keras.__version__)
maxX1=5000
minX1=1
maxX2=588.21
minX2=26.61
i=0
model=load_model('/home/mohammad/abides1/agent/examples/genrator_model.h5')
model.summary()

data=pd.read_csv('/home/mohammad/abides1/agent/examples/orderbook.csv')

scaler = joblib.load('/home/mohammad/abides1/agent/examples/my_dope_model.pkl') 

def getbestaskbid():
	var=randrange(10000)
	BA=data.iloc[:,0]
	BB=data.iloc[:,2]
	return BA[var], BB[var]

def generate_latent_points(latent_dim, n):
	x_input = randn(latent_dim * n)
	x_input = x_input.reshape(n, latent_dim)
	return x_input


class ExampleExperimentalAgentTemplate(SubscriptionAgent):
    """ Minimal working template for an experimental trading agent
    """
    def __init__(self, id, name, type, symbol, starting_cash, levels, subscription_freq, log_orders=False, random_state=None):
        """  Constructor for ExampleExperimentalAgentTemplate.

        :param id: Agent's ID as set in config
        :param name: Agent's human-readable name as set in config
        :param type: Agent's human-readable type as set in config, useful for grouping agents semantically
        :param symbol: Name of asset being traded
        :param starting_cash: Dollar amount of cash agent starts with.
        :param levels: Number of levels of orderbook to subscribe to
        :param subscription_freq: Frequency of orderbook updates subscribed to (in nanoseconds)
        :param log_orders: bool to decide if agent's individual actions logged to file.
        :param random_state: numpy RandomState object from which agent derives randomness
        """
        super().__init__(id, name, type, symbol, starting_cash, levels, subscription_freq, log_orders=log_orders, random_state=random_state)

        self.current_bids = None  # subscription to market data populates this list
        self.current_asks = None  # subscription to market data populates this list

    def wakeup(self, currentTime):
        """ Action to be taken by agent at each wakeup.

            :param currentTime: pd.Timestamp for current simulation time
        """
        super().wakeup(currentTime)
        self.setWakeup(currentTime + self.getWakeFrequency())

    def receiveMessage(self, currentTime, msg):
        """ Action taken when agent receives a message from the exchange

        :param currentTime: pd.Timestamp for current simulation time
        :param msg: message from exchange
        :return:
        """
        super().receiveMessage(currentTime, msg)  # receives subscription market data

    def getWakeFrequency(self):
        """ Set next wakeup time for agent. """
        return pd.Timedelta("1min")

    def placeLimitOrder(self, quantity, is_buy_order, limit_price):
        """ Place a limit order at the exchange.
          :param quantity (int):      order quantity
          :param is_buy_order (bool): True if Buy else False
          :param limit_price: price level at which to place a limit order
          :return:
        """
        super().placeLimitOrder(self.symbol, quantity, is_buy_order, limit_price)

    def placeMarketOrder(self, quantity, is_buy_order):
        """ Place a market order at the exchange.
          :param quantity (int):      order quantity
          :param is_buy_order (bool): True if Buy else False
          :return:
        """
        super().placeMarketOrder(self.symbol, quantity, is_buy_order)

    def cancelAllOrders(self):
        """ Cancels all resting limit orders placed by the experimental agent.
        """
        for _, order in self.orders.items():
            self.cancelOrder(order)


class ExampleExperimentalAgent(ExampleExperimentalAgentTemplate):

    def __init__(self, *args, wake_freq, order_size, short_window, long_window, **kwargs):
        """
        :param args: superclass args
        :param wake_freq: Frequency of wakeup -- str to be parsed by pd.Timedelta
        :param order_size: size of orders to place
        :param short_window: length of mid price short moving average window -- str to be parsed by pd.Timedelta
        :param long_window: length of mid price long moving average window -- str to be parsed by pd.Timedelta
        :param kwargs: superclass kwargs
        """
        super().__init__(*args, **kwargs)
        self.wake_freq = wake_freq
        self.order_size = order_size
        self.short_window = short_window
        self.long_window = long_window
        self.i=1
        self.mid_price_history = pd.DataFrame(columns=['mid_price'], index=pd.to_datetime([]))                        #empty array

    def getCurrentMidPrice(self):                                                                                     #if any change here
        """ Retrieve mid price from most recent subscription data.

        :return:
        """


        try:
        
            if self.i<8:
            	best_bid = self.current_bids[0][0]
            	best_ask = self.current_asks[0][0]
            	self.i=self.i+1
            else:
            	aaa,bbb=getbestaskbid()
            	best_bid=bbb
            	best_ask=aaa
            	self.i=self.i+1
            return round((best_ask + best_bid) / 2)
        except (TypeError, IndexError):
            return None

    def receiveMessage(self, currentTime, msg):
    	try:
    		super().receiveMessage(currentTime, msg)
    		#print(currentTime)
    		self.mid_price_history = self.mid_price_history.append(pd.Series({'mid_price': self.getCurrentMidPrice()}, 		name=currentTime))
    		self.mid_price_history.dropna(inplace=True)
    		#print(self.mid_price_history)
    	except :
            print("NN")
    	'''
        
        """ Action taken when agent receives a message from the exchange -- action here is for agent to update internal
            log of most recently observed mid-price.

        :param currentTime: pd.Timestamp for current simulation time
        :param msg: message from exchange
        :return:
        """
        # receives subscription market data
      	'''
    def computeMidPriceMovingAverages(self):
    	try:
    		short_moving_avg = self.mid_price_history.rolling(self.short_window).mean().iloc[-1]['mid_price']
    		long_moving_avg = self.mid_price_history.rolling(self.long_window).mean().iloc[-1]['mid_price']
    		return short_moving_avg, long_moving_avg
    	except IndexError:
    		return None, None

    	'''
        """ Returns the short-window and long-window moving averages of mid price.
        :return:
        """
        '''
    def wakeup(self, currentTime):
    	
    	latent_dim = 4
    	n=1
    	noise=np.random.normal(0,1,(n,latent_dim))
    	gd=model.predict(noise)
    	gd2=scaler.inverse_transform(gd)
    	gddf=pd.DataFrame(data=gd2,columns=['size','price','direction'])
    	size=gddf['size']
    	price=gddf['price']
    	direction=gddf['direction']
    	direction[direction<0]=-1
    	direction[direction>0]=1
    	Var1=size
    	Var2=price
    	Var3=direction
    	
    	
    	#latent_points = generate_latent_points(latent_dim, n)
    	#X = model.predict(latent_points)
    	
    	#Var1 = (X[:,0]) * (maxX1 - minX1) + minX1
    	#Var1=np.abs(Var1)
    	#Var1=np.around(Var1)
    	#print(Var1)
    	
    	#Var2 = X[:,1] * (maxX2 - minX2) + minX2
    	#Var2=np.abs(Var2)
    	#Var2=Var2*10000
    	#print(Var2)
    	
    	#Var3 = X[:,2]
    	#Var3[Var3 < 0] = -1
    	#Var3[Var3 > 0] = 1
    	#print(Var3)
    	
    	
    	volume=Var1
    	self.order_size = int(volume)
    	price=Var2
    	direction=Var3
    		
    	super().wakeup(currentTime)
    	short_moving_avg, long_moving_avg = self.computeMidPriceMovingAverages()
    	if short_moving_avg is not None and long_moving_avg is not None:
    		#if direction==-1:
    		#	short_moving_avg=20
    		#	long_moving_avg =10
    		#if direction==1:
    		#	short_moving_avg=10
    		#	long_moving_avg =20
    		if short_moving_avg > long_moving_avg:
    			self.placeMarketOrder(self.order_size, 0)
    		elif short_moving_avg < long_moving_avg:
    			self.placeMarketOrder(self.order_size, 1)
    def getWakeFrequency(self):
        """ Set next wakeup time for agent. """
        return pd.Timedelta(self.wake_freq)



