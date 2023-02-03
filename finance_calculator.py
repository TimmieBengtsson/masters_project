import numpy as np
import math
import utils


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
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("Accuracy: {:5.2f}%".format(100 * accuracy))
    print('')
    print("TPR: {:5.2f}%".format(100 * recall))
    print("TNR: {:5.2f}%".format(100 * specificity))
    print("PPV: {:5.2f}%".format(100 * precision))
    print("NPV: {:5.2f}%".format(100 * neg_precision))
    print('')


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


def sharpe_ratio2(portfolio_values, rf):
    total_real_return = gross_return(portfolio_values) -rf
    nbr_of_years_held_investment = len(portfolio_values)/252
    total_real_return_annualized = pow(1 + total_real_return, 1/nbr_of_years_held_investment) -1
    portfolio_volatility_yearly = portfolio_yearly_standard_deviation(portfolio_values)
    return total_real_return_annualized / portfolio_volatility_yearly


def sortino_ratio(returns, trading_days_year, rf):
    mean = (returns.mean() * trading_days_year) -rf
    std_neg = returns[returns<0].std() * np.sqrt(trading_days_year)
    return mean / std_neg


def maximal_drawdown(portfolio_value):
    running_max = np.maximum.accumulate(portfolio_value)
    drawdown = (running_max - portfolio_value) / running_max
    return np.max(drawdown) *-1


def returns(prices):
    returns = []
    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(daily_return)
    return np.array(returns)


def gross_return(pv):
    return (pv[-1]-pv[0]) / pv[0]


def portfolio_yearly_standard_deviation(portfolio_values):
    every_fifth_value = portfolio_values[0::5]
    weekly_returns = returns(every_fifth_value)
    yearly_variance = weekly_returns.var() * 52
    yearly_standard_deviation = np.sqrt(yearly_variance)
    return yearly_standard_deviation


def portfolio_financial_stats(portfolio_value, one_year_of_returns):
    RISK_FREE_RATE = 0.02
    portfolio_returns = returns(portfolio_value)
    sharpe = sharpe_ratio(portfolio_returns, one_year_of_returns, RISK_FREE_RATE)
    sortino = sortino_ratio(portfolio_returns, one_year_of_returns, RISK_FREE_RATE)
    print("Sharpe-ratio: {:5.2f}".format(sharpe))
    print("Sortino-ratio: {:5.2f}".format(sortino))
    portfolio_gross_return = gross_return(portfolio_value)
    print("Gross return: {:5.2f}%".format(100 * portfolio_gross_return))
    portfolio_maximal_drawdown = maximal_drawdown(portfolio_value)
    print("Maximal Drawdown: {:5.2f}%".format(100 * portfolio_maximal_drawdown))
    save_fin_stats_variables(sharpe, sortino, portfolio_gross_return, portfolio_maximal_drawdown)
    

def save_fin_stats_variables(sharpe, sortino, gross_return, maximal_drawdown):
    NAME = utils.model_loader_evaluate_name()
    utils.variable_save(f'{NAME}_sharpe-ratio', sharpe)
    utils.variable_save(f'{NAME}_sortino-ratio', sortino)
    utils.variable_save(f'{NAME}_gross-return', gross_return)
    utils.variable_save(f'{NAME}_maximal-drawdown', maximal_drawdown)
