import pandas as pd

df1 = pd.read_json('2022-09-12 22:50:41.928440.json')                      
df2 = pd.read_json('2022-09-12 23:29:49.672279.json')                      
df1['long_profit'] = (df1['profit_ratio'].multiply(100)).round(2)
df1['short_profit'] = (df2['profit_ratio'].multiply(100)).round(2)

df = df1.iloc[:,-7:]
df.loc[(df.long_profit > 0) & (df.short_profit < 0), 'label'] = 'long'
df.loc[(df.long_profit < 0) & (df.short_profit > 0), 'label'] = 'short'
df.loc[(df.long_profit < 0) & (df.short_profit < 0), 'label'] = 'None'

