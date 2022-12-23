import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
import matplotlib.pyplot as plt

from collections import deque
import random
from sklearn import preprocessing


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def create_model(train_x):
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
    # Compile model
    model.compile(optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def preprocess_df(df, SEQ_LEN):
    df = df.drop(columns="t+1")
    for col in df.columns:
        if col != "target": 
            df[col] = df[col].pct_change() 
            df.dropna(inplace=True)  
            df[col] = preprocessing.scale(df[col].values) 
            
    df.dropna(inplace=True) 

    sequential_data = [] 
    prev_days = deque(maxlen=SEQ_LEN) 
    for i in df.values:
        prev_days.append([n for n in i[:-1]]) 
        if len(prev_days) == SEQ_LEN:  
            sequential_data.append([np.array(prev_days), i[-1]])  
    random.shuffle(sequential_data)

    buys = []  
    sells = [] 
    for seq, target in sequential_data:
        if target == 0:  # if it's a sell
            sells.append([seq, target])  
        elif target == 1: 
            buys.append([seq, target])  # it's a buy
    random.shuffle(buys)  
    random.shuffle(sells) 

    lower = min(len(buys), len(sells))  # what's the shorter length?
    buys = buys[:lower]  
    sells = sells[:lower]

    sequential_data = buys+sells 
    random.shuffle(sequential_data) 

    X = []
    y = []
    for seq, target in sequential_data:  
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels

    return np.array(X).astype("float32"), np.array(y)


def split(df):
    times = sorted(df.index.values)  
    last_10pct = sorted(df.index.values)[-int(0.10*len(times))] 
    df_train = df[(df.index < last_10pct)] 
    df_val = df[(df.index >= last_10pct)] 
    return df_train, df_val
    

def percentage_change(t,t1):
    return (t1-t)/t

# old version, will remove eventually if not needed.
def format_to_input_old(df):
    df = df.pct_change()
    df.dropna(inplace=True)
    df = preprocessing.scale(df.values)
    return df


def format_to_input(df):
    for col in df.columns:
        df[col] = df[col].pct_change() 
        df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)  
    return df


def crypto_data_reader(ratios):
    df_return = pd.DataFrame()
    for ratio in ratios: 
        ratio = ratio.split('.csv')[0] 
        dataset = f'data/crypto_data/{ratio}.csv'
        df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume']) 
        df.rename(columns={"close": f"{ratio}_close"}, inplace=True)
        df.set_index("time", inplace=True)
        df = df[[f"{ratio}_close"]] 
        if len(df_return)==0:  
            df_return = df  
        else:
            df_return = df_return.join(df)
    
    df_return.fillna(method="ffill", inplace=True)
    df_return.dropna(inplace=True)
    return df_return


def plot_history_data(history):
    # summarize history for accuracy
    fig1, axs1 = plt.subplots(1,1, figsize=(10,5))
    fig1 = plt.plot(history['accuracy'])
    fig1 = plt.plot(history['val_accuracy'])
    axs1 = plt.title('Model accuracy')
    axs1 = plt.ylabel('Accuracy')
    axs1 = plt.xlabel('Epochs')
    axs1 = plt.legend(['train', 'Validation'], loc='upper left')

    # summarize history for loss
    fig2, axs2 = plt.subplots(1,1, figsize=(10,5))
    fig1 = plt.plot(history['loss'])
    fig1 = plt.plot(history['val_loss'])
    axs1 = plt.title('Model loss')
    axs1 = plt.ylabel('Loss')
    axs1 = plt.xlabel('Epochs')
    axs1 = plt.legend(['Train', 'Validation'], loc='upper left')

    return fig1, axs1, fig2, axs2



def get_rpp(val_df, seq, asset_name, model):
    df_rpp = pd.DataFrame(columns=('r', 'pred[0]', 'pred[1]'))

    for t in range(seq, len(val_df)-1):
        value_t = val_df.iloc[t][asset_name]
        value_t1 = val_df.iloc[t+1][asset_name]
        r = percentage_change(value_t,value_t1)

        values = val_df.iloc[t-seq:t,].copy()
        values = format_to_input(values)
        values = values.values.reshape(1,seq-1,len(val_df.columns))
        pred = model.predict(values)[-1] #predict t+1 and retrive last prediction

        df_last = pd.DataFrame([(r, pred[0], pred[1])], columns=('r', 'pred[0]', 'pred[1]'))
        df_rpp = pd.concat([df_rpp, df_last])
        print(t)

    df_rpp = df_rpp.reset_index(drop=True)
    return df_rpp



def get_portfolio_value_trading(df_rpp, treshhold):
    portfolio_value = np.zeros(len(df_rpp))
    cash = 100
    for t in range(0, len(df_rpp)):
        portfolio_value[t] = cash
        pred = df_rpp.loc[t,'pred[0]']
        r = df_rpp.loc[t,'r']
        if pred < treshhold:
            cash = cash * (1+r)
        else:
            cash = cash * (1-r)
    
    return portfolio_value

def get_portfolio_value_hold(df_rpp):
    portfolio_value = np.zeros(len(df_rpp))
    cash = 100
    for t in range(0, len(df_rpp)):
        portfolio_value[t] = cash
        r = df_rpp.loc[t,'r']
        cash = cash * (1+r)

    return portfolio_value


def get_statistics_trading(df_rpp, treshhold):
    tp, fp, tn, fn = 0, 0, 0, 0
    for t in range(0, len(df_rpp)):
        pred = df_rpp.loc[t,'pred[0]']
        r = df_rpp.loc[t,'r']
        if pred < treshhold:
            if r >= 0:
                tp +=1
            if r < 0:
                fp +=1
        if pred > treshhold:
            if r >= 0:
                fn +=1
            if r < 0:
                tn +=1
    return tp, fp, tn, fn


def sharpe_ratio(returns, trading_days_year, rf):
    mean = (returns.mean() * trading_days_year) -rf
    sigma = returns.std() * np.sqrt(trading_days_year)
    return mean / sigma

def sortino_ratio(returns, trading_days_year, rf):
    mean = (returns.mean() * trading_days_year) -rf
    std_neg = returns[returns<0].std() * np.sqrt(trading_days_year)
    return mean / std_neg

def max_drawdown(returns):
    comp_ret = (returns+1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()

def get_daily_returns(prices):
    returns = []
    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(daily_return)
    return np.array(returns)


def get_gross_return(pv):
    return (pv[-1]-pv[0]) / pv[0]




