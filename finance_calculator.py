import numpy as np


def portfolio_value_neural(df_rpp, treshhold):
    portfolio_value = np.zeros(len(df_rpp))
    cash = 100
    for t in range(0, len(df_rpp)):
        portfolio_value[t] = cash
        pred = df_rpp.loc[t, 'probability_up']
        r = df_rpp.loc[t,'r']
        if pred >= treshhold:
            cash = cash * (1+r)
        else:
            cash = cash * (1-r)
    return portfolio_value


def portfolio_value_hold(df_rpp):
    portfolio_value = np.zeros(len(df_rpp))
    cash = 100
    for t in range(0, len(df_rpp)):
        portfolio_value[t] = cash
        r = df_rpp.loc[t,'r']
        cash = cash * (1+r)
    return portfolio_value


def portfolio_value_naive(df_rpp):
    portfolio_value = np.zeros(len(df_rpp))
    cash = 100
    for t in range(0, len(df_rpp)):
        portfolio_value[t] = cash
        if t == 0:
            continue
        r_previous = df_rpp.loc[t-1,'r']
        r = df_rpp.loc[t,'r']
        if r_previous >= 0:
            cash = cash * (1+r)
        else:
            cash = cash * (1-r)
    return portfolio_value

def portfolio_value_trend(df_rpp, nbr_previous_days):
    portfolio_value = np.zeros(len(df_rpp))
    cash = 100
    for t in range(0, len(df_rpp)):
        portfolio_value[t] = cash
        if t < nbr_previous_days:
            continue
        sum_previous_days = 0
        for i in range(1, nbr_previous_days):
            r_previous = df_rpp.loc[t-i,'r']
            sum_previous_days = sum_previous_days + r_previous
        previous_days_average = sum_previous_days / nbr_previous_days    
        r = df_rpp.loc[t,'r']
        if previous_days_average >= 0:
            cash = cash * (1+r)
        else:
            cash = cash * (1-r)        
    return portfolio_value


def confusion_stats_printer(tp, fp, tn, fn):
    print(f'TP: {tp}'); print(f'FP: {fp}'); print(f'TN: {tn}'); print(f'FN: {fn}')
    recall = tp / (tp + fn) #TP / P
    specificity = tn / (tn + fp) #TN / N
    precision = tp / (tp + fp)
    neg_precision = tn / (tn + fn)
    print("True positive rate: {:5.2f}%".format(100 * recall))
    print("True negative rate: {:5.2f}%".format(100 * specificity))
    print("Positive predictive value: {:5.2f}%".format(100 * precision))
    print("Negative predictive value: {:5.2f}%".format(100 * neg_precision))


def confusion_stats_trend(df_rpp, nbr_previous_days):
    tp, fp, tn, fn = 0, 0, 0, 0
    for t in range(0, len(df_rpp)):
        if t < nbr_previous_days:
            continue
        sum_previous_days = 0
        for i in range(1, nbr_previous_days):
            r_previous = df_rpp.loc[t-i,'r']
            sum_previous_days = sum_previous_days + r_previous
        previous_days_average = sum_previous_days / nbr_previous_days    
        r = df_rpp.loc[t,'r']
        if previous_days_average >= 0:
            if r >= 0:
                tp +=1
            if r < 0:
                fp +=1
        if previous_days_average < 0:
            if r < 0:
                tn +=1
            if r >= 0:
                fn +=1
    print('')
    print('TREND STATS:')
    confusion_stats_printer(tp, fp, tn, fn)


def confusion_stats_naive(df_rpp):
    tp, fp, tn, fn = 0, 0, 0, 0
    for t in range(0, len(df_rpp)):
        if t == 0:
            continue
        r_previous = df_rpp.loc[t-1,'r']
        r = df_rpp.loc[t,'r']
        if r_previous >= 0:
            if r >= 0:
                tp +=1
            if r < 0:
                fp +=1
        if r_previous < 0:
            if r < 0:
                tn +=1
            if r >= 0:
                fn +=1
    print('')
    print('NAIVE STATS:')
    confusion_stats_printer(tp, fp, tn, fn)
    

def confusion_stats_neural(df_rpp, treshhold):
    tp, fp, tn, fn = 0, 0, 0, 0
    for t in range(0, len(df_rpp)):
        pred = df_rpp.loc[t, 'probability_up']
        r = df_rpp.loc[t,'r']
        if pred >= treshhold:
            if r >= 0:
                tp +=1
            if r < 0:
                fp +=1
        if pred < treshhold:
            if r < 0:
                tn +=1
            if r >= 0:
                fn +=1
    print('')
    print('NEURAL STATS:')
    confusion_stats_printer(tp, fp, tn, fn)


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


def daily_returns(prices):
    returns = []
    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(daily_return)
    return np.array(returns)


def gross_return(pv):
    return (pv[-1]-pv[0]) / pv[0]
