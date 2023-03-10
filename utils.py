import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import finance_calculator
import csv
import os

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def plot_history_data(history):
    # summarize history for accuracy
    fig1, axs1 = plt.subplots(1,1)
    fig1 = plt.plot(history['accuracy'],  color='firebrick', linestyle='-')
    fig1 = plt.plot(history['val_accuracy'],  color='darkblue', linestyle='-')
    axs1 = plt.title('Model Accuracy', )
    axs1 = plt.ylabel('Accuracy')
    axs1 = plt.xlabel('Epochs')
    axs1 = plt.legend(['Train', 'Validation'], loc='upper left')
    # summarize history for loss
    fig2, axs2 = plt.subplots(1,1)
    fig1 = plt.plot(history['loss'],  color='firebrick', linestyle='-')
    fig1 = plt.plot(history['val_loss'],  color='darkblue', linestyle='-')
    axs1 = plt.title('Model Loss')
    axs1 = plt.ylabel('Loss')
    axs1 = plt.xlabel('Epochs')
    axs1 = plt.legend(['Train', 'Validation'], loc='upper left')
    return fig1, axs1, fig2, axs2

def plot_history_data_accuracy(history):
    # summarize history for accuracy
    fig1, axs1 = plt.subplots(1,1)
    fig1 = plt.plot(history['accuracy'],  color='firebrick', linestyle='-', linewidth='1')
    fig1 = plt.plot(history['val_accuracy'],  color='darkblue', linestyle='-', linewidth='1')
    axs1 = plt.title('Model Accuracy', )
    axs1 = plt.ylabel('Accuracy')
    axs1 = plt.xlabel('Epochs')
    axs1 = plt.legend(['Train', 'Validation'], loc='upper left')
    return fig1, axs1

def plot_history_data_loss(history):
    # summarize history for loss
    fig1, axs1 = plt.subplots(1,1)
    fig1 = plt.plot(history['loss'],  color='firebrick', linestyle='-', linewidth='1')
    fig1 = plt.plot(history['val_loss'],  color='darkblue', linestyle='-', linewidth='1')
    axs1 = plt.title('Model Loss')
    axs1 = plt.ylabel('Loss')
    axs1 = plt.xlabel('Epochs')
    axs1 = plt.legend(['Train', 'Validation'], loc='upper left')
    return fig1, axs1


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
    df_rpp = pd.DataFrame(columns=('r', 'probability_up', 'probability_down'))
    for t in range(seq, len(val_df)-1):
        value_t = val_df.iloc[t][asset_name]
        value_t1 = val_df.iloc[t+1][asset_name]
        r = percentage_change(value_t,value_t1)
        values = val_df.iloc[t-seq:t,].copy()
        values = format_to_input(values)
        values = values.values.reshape(1,seq-1,len(val_df.columns))
        pred = model.predict_on_batch(values)[-1] #predict t+1 and retrive last prediction
        print(pred)
        df_last = pd.DataFrame([(r, pred[0], pred[1])], columns=('r', 'probability_up', 'probability_down'))
        df_rpp = pd.concat([df_rpp, df_last])
    df_rpp = df_rpp.reset_index(drop=True)
    return df_rpp


def getReturnAndPred_sigmoidModel(val_df, seq, asset_name, model):
    df_rp = pd.DataFrame(columns=('r', 'probability_up'))
    for t in range(seq, len(val_df)-1):
        value_t = val_df.iloc[t][asset_name]
        value_t1 = val_df.iloc[t+1][asset_name]
        r = percentage_change(value_t,value_t1)
        values = val_df.iloc[t-seq:t,].copy()
        values = format_to_input(values)
        values = values.values.reshape(1,seq-1,len(val_df.columns))
        pred = model.predict_on_batch(values)[-1] #predict t+1 and retrive last prediction
        if t % 75 == 0:
            print(f'Pred #{t}: {pred}')
        df_last = pd.DataFrame([(r, pred[0])], columns=('r', 'probability_up'))
        df_rp = pd.concat([df_rp, df_last])
    df_rp = df_rp.reset_index(drop=True)
    return df_rp


def model_loader_train_name(NAME=None):
    if NAME is None:
        with open('models/last_trained_model_name.txt') as f:
            NAME = f.readline()
            print(f'Model name from latest training was loaded.\nLast trained model name: {NAME}')
            return NAME
    return NAME

def model_loader_evaluate_name(NAME=None):
    if NAME is None:
        with open('models/last_evaluated_model_name.txt') as f:
            NAME = f.readline()
            print(f'Model name from latest evaluation was loaded.\nLast evaluated model name: {NAME}')
            return NAME
    return NAME


def set_last_trained_model_name(NAME):
    with open('models/last_trained_model_name.txt', 'w') as f:
        f.write(NAME)
        print(f'Last trained model name is set to: {NAME}')

def set_last_evaluated_model_name(NAME):
    with open('models/last_evaluated_model_name.txt', 'w') as f:
        f.write(NAME)
        print(f'Last evaluated model name is set to: {NAME}')


def variable_save(key, value):
    dict_var = {}
    file_path = os.path.join(os.getcwd(), "variables.dat")
    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass
    dict_var[key] = value
    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")


