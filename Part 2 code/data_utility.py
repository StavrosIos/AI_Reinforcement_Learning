
"""
Script to load the time series data. It prforms normalization on the <Volumes> feature, in a way that is more appropriate for Neural networks
The escript also chnages absolute prices to relative ones.
"""

import numpy as np
import collections
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

def relative_prices(stock):

    relative_high = (stock.high - stock.open) / stock.open  #percentage changes in the stock price
    relative_low = (stock.low - stock.open) / stock.open
    relative_close = (stock.close - stock.open) / stock.open
    return Prices(open=stock.open, high=relative_high, low=relative_low, close=relative_close, volume=stock.volume)
 

def filter_function(row):
      output_row=None
      if row[0]==row[1]==row[2]==row[3]:
            row[0]=0
            row[1]=0
            row[2]=0
            row[3]=0
            row[4]=0
            output_row=row
      else:
            output_row=row.copy()
            
      return output_row      



def load_stock(csv_file_path):
      print("Loading data...")
      #load test data and absolute closing prices to plot decisions
      datum=pd.read_csv(csv_file_path)
      
      datum.drop(['timestamp'],axis=1,inplace=True)
      
      final_datum=datum.apply(filter_function,axis=1)

      final_datum = final_datum.replace(0, np.nan)
      final_datum = final_datum.dropna(how='all', axis=0)

      std_volumes=  np.float32(scaler.fit_transform(final_datum['<VOL>'].values.reshape(-1,1))) #float 64  to float 32
      
      stocks=Prices(open=np.array(final_datum['<OPEN>'].values, dtype=np.float32),
                  high=np.array(final_datum['<HIGH>'].values, dtype=np.float32),
                  low=np.array(final_datum['<LOW>'].values, dtype=np.float32),
                  close=np.array(final_datum['<CLOSE>'].values,dtype=np.float32),
                  volume=np.array(std_volumes, dtype=np.float32))
      
      print("Data loaded !")
      return relative_prices(stocks), final_datum['<CLOSE>']



def load_stock_absolute_volumes(csv_file_path):
      print("Loading data...")
      #load test data and absolute closing prices to plot decisions
      datum=pd.read_csv(csv_file_path)
      
      datum.drop(['timestamp'],axis=1,inplace=True)
      
      final_datum=datum.apply(filter_function,axis=1)

      final_datum = final_datum.replace(0, np.nan)
      final_datum = final_datum.dropna(how='all', axis=0)

      #std_volumes=  np.float32(scaler.fit_transform(final_datum['<VOL>'].values.reshape(-1,1))) #float 64  to float 32
      
      stocks=Prices(open=np.array(final_datum['<OPEN>'].values, dtype=np.float32),
                  high=np.array(final_datum['<HIGH>'].values, dtype=np.float32),
                  low=np.array(final_datum['<LOW>'].values, dtype=np.float32),
                  close=np.array(final_datum['<CLOSE>'].values,dtype=np.float32),
                  volume=np.array(final_datum['<VOL>'].values, dtype=np.float32))
      
      print("Data loaded !")
      return relative_prices(stocks), final_datum['<CLOSE>']




