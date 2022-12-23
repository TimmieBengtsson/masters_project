import pandas as pd
import numpy as np
import math
import tensorflow as tf

def credit_data_reader():
    df = pd.read_csv('data/shb_data/terminspriser_shb.csv', delimiter=';')
    df = df[:-1]
    df = df[['DATE', 'RX1 Comdty', 'TY1 Comdty', 'IK1 Comdty', 'OE1 Comdty', 'DU1 Comdty']]
    df = df.stack().str.replace(',', '.').unstack()
    df = df.astype(
        {
            'RX1 Comdty': 'float',
            'TY1 Comdty': 'float',
            'IK1 Comdty': 'float',
            'OE1 Comdty': 'float',
            'DU1 Comdty': 'float'
        }
    )
    rx1 = df[['RX1 Comdty']]
    rx1 = rx1[rx1['RX1 Comdty'].notna()]
    ty1 = df[['TY1 Comdty']]
    ty1 = ty1[ty1['TY1 Comdty'].notna()]
    ik1 = df[['IK1 Comdty']]
    ik1 = ik1[ik1['IK1 Comdty'].notna()]
    oe1 = df[['OE1 Comdty']]
    oe1 = oe1[oe1['OE1 Comdty'].notna()]
    du1 = df[['DU1 Comdty']]
    du1 = du1[du1['DU1 Comdty'].notna()]
    return rx1, ty1, ik1, oe1, du1

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
   x_data, y_data = [], []
   for i in range(look_back, len(dataset)-1):
      a = dataset[i-look_back:i, 0]
      x_data.append(a)
      b = dataset[i:i+1, 0]
      y_data.append(b)
   return np.array(x_data), np.array(y_data)

# convert an array of values into a dataset matrix (with binary labels)
def create_dataset_binarylabel(dataset, look_back):
   x_data, y_data = [], []
   for i in range(look_back, len(dataset)-1):
      a = dataset[i-look_back:i, 0]
      x_data.append(a)
      b = dataset[i:i+1, 0]
      if(b > 0):
        y_data.append(1)
      else:
        y_data.append(0)
   return np.array(x_data), np.array(y_data)

# Special activation function
def modified_sigmoid(x):
    return 2 * (1 / (1 + tf.math.exp(x))) - 1

# Load airline data and calc returns 
def fetch_airline_returns():
    df = pd.read_csv('data/airline-passengers.csv', usecols=[1], engine='python')
    df_returns = df.pct_change()[1:]
    returns = df_returns.to_numpy()
    returns = returns.astype('float32')# more suitable for NNs
    return returns

# split into train and test sets
def split_into_tvt(dataset):
    train_size = int(len(dataset) * 0.6)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - (train_size+val_size)
    train, val, test = dataset[0:train_size,:], dataset[train_size:train_size+val_size,:], dataset[train_size+val_size:len(dataset),:]
    return train, val, test

def get_returns(df):
    df = df.pct_change()[1:]
    df = df.to_numpy()
    df = df.astype('float32')
    return df

def credit_data_reader_returns():
    rx1, ty1, ik1, oe1, du1 = credit_data_reader()
    rx1 = get_returns(rx1)
    ty1 = get_returns(ty1)
    ik1 = get_returns(ik1)
    oe1 = get_returns(oe1)
    du1 = get_returns(du1)
    return rx1, ty1, ik1, oe1, du1






