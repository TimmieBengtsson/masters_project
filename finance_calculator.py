import numpy as np


def portfolio_value_trading(df_rpp, treshhold):
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


def portfolio_value_hold(df_rpp):
    portfolio_value = np.zeros(len(df_rpp))
    cash = 100
    for t in range(0, len(df_rpp)):
        portfolio_value[t] = cash
        r = df_rpp.loc[t,'r']
        cash = cash * (1+r)
    return portfolio_value


def confusion_stats_trading(df_rpp, treshhold):
    tp, fp, tn, fn = 0, 0, 0, 0
    for t in range(0, len(df_rpp)):
        pred = df_rpp.loc[t,'pred[0]']
        r = df_rpp.loc[t,'r']
        if pred <= treshhold:
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


def daily_returns(prices):
    returns = []
    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(daily_return)
    return np.array(returns)


def gross_return(pv):
    return (pv[-1]-pv[0]) / pv[0]
