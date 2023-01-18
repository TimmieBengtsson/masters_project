import random
import numpy as np
from collections import deque
from sklearn import preprocessing


def preprocess(df, SEQ_LEN):
    df = df.drop(columns="t+1")
    for col in df.columns:
        if col != "target": 
            df[col] = df[col].pct_change() 
            df.dropna(inplace=True)  
            df[col] = preprocessing.scale(df[col].values) 
            
    df.dropna(inplace=True) 

    seq_data = [] 
    prev_days = deque(maxlen=SEQ_LEN) 
    for i in df.values:
        prev_days.append([n for n in i[:-1]]) 
        if len(prev_days) == SEQ_LEN:  
            seq_data.append([np.array(prev_days), i[-1]])  
    random.shuffle(seq_data)

    ups, downs = [], []
    for seq, target in seq_data:
        if target == 0:  # if it's a sell
            downs.append([seq, target])  
        elif target == 1: 
            ups.append([seq, target])  # it's a buy
    random.shuffle(ups)  
    random.shuffle(downs) 
    print('Ups: ', len(ups))
    print('Downs: ', len(downs))

    lower = min(len(ups), len(downs))  # what's the shorter length?
    ups, downs = ups[:lower], downs[:lower]
    seq_data = ups+downs 
    random.shuffle(seq_data) 

    X, y = [], []
    for seq, target in seq_data:  
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels

    return np.array(X).astype("float32"), np.array(y)


def splitter(df, cutoff):
    times = sorted(df.index.values)  
    last_10pct = sorted(df.index.values)[-int(cutoff*len(times))] 
    df_first_sequence = df[(df.index < last_10pct)] 
    df_last_sequence = df[(df.index >= last_10pct)] 
    return df_first_sequence, df_last_sequence