import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid", {'grid.linestyle': '--'})
seq_col_brew = sns.color_palette("flag_r", 4)
sns.set_palette(seq_col_brew)
plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams["axes.titlesize"] = 17
plt.rcParams['savefig.dpi'] = 1200

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def plot_history_data(history):
    # summarize history for accuracy
    fig1, axs1 = plt.subplots(1,1)
    fig1 = plt.plot(history['accuracy'])
    fig1 = plt.plot(history['val_accuracy'])
    axs1 = plt.title('Model Accuracy', )
    axs1 = plt.ylabel('Accuracy')
    axs1 = plt.xlabel('Epochs')
    axs1 = plt.legend(['Train', 'Validation'], loc='upper left')

    # summarize history for loss
    fig2, axs2 = plt.subplots(1,1)
    fig1 = plt.plot(history['loss'])
    fig1 = plt.plot(history['val_loss'])
    axs1 = plt.title('Model Loss')
    axs1 = plt.ylabel('Loss')
    axs1 = plt.xlabel('Epochs')
    axs1 = plt.legend(['Train', 'Validation'], loc='upper left')
    return fig1, axs1, fig2, axs2

# Helper method
def format_to_input(df):
    for col in df.columns:
        df[col] = df[col].pct_change() 
        df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)  
    return df

# Helper method
def percentage_change(t,t1):
    return (t1-t)/t

def getReturnAndPreds_softmaxModel(val_df, seq, asset_name, model):
    df_rpp = pd.DataFrame(columns=('r', 'pred[0]', 'pred[1]'))

    for t in range(seq, len(val_df)-1):
        value_t = val_df.iloc[t][asset_name]
        value_t1 = val_df.iloc[t+1][asset_name]
        r = percentage_change(value_t,value_t1)

        values = val_df.iloc[t-seq:t,].copy()
        values = format_to_input(values)
        values = values.values.reshape(1,seq-1,len(val_df.columns))
        pred = model.predict_on_batch(values)[-1] #predict t+1 and retrive last prediction
        print(pred)
        df_last = pd.DataFrame([(r, pred[0], pred[1])], columns=('r', 'pred[0]', 'pred[1]'))
        df_rpp = pd.concat([df_rpp, df_last])

    df_rpp = df_rpp.reset_index(drop=True)
    return df_rpp

def getReturnAndPred_sigmoidModel(val_df, seq, asset_name, model):
    df_rp = pd.DataFrame(columns=('r', 'pred[0]'))

    for t in range(seq, len(val_df)-1):
        value_t = val_df.iloc[t][asset_name]
        value_t1 = val_df.iloc[t+1][asset_name]
        r = percentage_change(value_t,value_t1)

        values = val_df.iloc[t-seq:t,].copy()
        values = format_to_input(values)
        values = values.values.reshape(1,seq-1,len(val_df.columns))
        pred = model.predict_on_batch(values)[-1] #predict t+1 and retrive last prediction
        print(pred)
        df_last = pd.DataFrame([(r, pred[0])], columns=('r', 'pred[0]'))
        df_rp = pd.concat([df_rp, df_last])

    df_rp = df_rp.reset_index(drop=True)
    return df_rp


