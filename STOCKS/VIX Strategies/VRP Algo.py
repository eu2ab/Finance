from urllib.request import urlopen
import pandas as pd
from pandas.io.data import DataReader
import datetime as DT
import numpy as np

#http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html for more visualizations

def historical_close(ticker,start,end): #datetime(2000, 1, 1) ***(yyyy, m, d) format****
    df = DataReader(ticker, 'yahoo', start, end)
    return df['Adj Close']

def historical_prices(ticker,start,end): #datetime(2000, 1, 1) ***(yyyy, m, d) format****
    df = DataReader(ticker, 'yahoo', start, end)
    return df

#set parameters for query
N= 150 #number of days into the past
end = DT.date.today()
start = end  - DT.timedelta(days=N)
stocks = ['^VIX','^VXV','XIV','VXX']
#import securities
df = historical_close(stocks,start,end)

#build dictionary
dict = pd.Series.to_dict(df)
#specific stocks
VIX = dict['^VIX'] #VIX
VXV = dict['^VXV'] #VXV
XIV = dict['XIV'] #XIV
VXX = dict['VXX'] #VXX
spread_3m = VXV - VIX #spread between 3m and spot
Vratio = VXV/VIX #above 1 = contango, below 1 = backwardation
slnVIX = np.std(np.log(VIX)) #standard deviation of log of VIX


#graphs
# Vratio.plot() #graph of Vratio
df=pd.concat([VXV,VIX],axis=1) #graph of VXV and VIX
df.plot(legend=True) #include legend
df.plot(subplots=True) #using subplots
df.plot.area(stacked=False)

#using a secondary y axis
VIX.plot()
Vratio.plot(secondary_y=True) #must be done after first graph



pd.rolling_corr(VXX,XIV,15) #rolling correlation with 15 day window
df.corr() #for correlation matrix of all stocks in df portfolio