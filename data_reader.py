import pandas as pd


def crypto(ratios):
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


def credit():
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