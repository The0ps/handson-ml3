from copy import deepcopy
import pandas as pd

# df3 = pd.read_json('2022-09-23 01:49:06.636107.json')
# df1 = pd.read_json('2022-09-12 22:50:41.928440.json')                      
# df2 = pd.read_json('2022-09-12 23:29:49.672279.json') MA strategy data
df1 = pd.read_json('2022-09-23 00:57:43.780634.json')                      
df2 = pd.read_json('2022-09-23 01:24:33.590226.json')                      
df1['long_profit'] = (df1['profit_ratio'].multiply(100)).round(2)
df1['short_profit'] = (df2['profit_ratio'].multiply(100)).round(2)

# df = df1[df.long_profit.notna()]
# df[df.long_profit.notna()]
df = df1
df.loc[(df.long_profit > 0) & (df.short_profit < 0), 'label'] = 'long'
df.loc[(df.long_profit < 0) & (df.short_profit > 0), 'label'] = 'short'
df.loc[(~df['label'].isin(['long', 'short'])) & (df.long_profit.notna()), 'label'] = 'hold'
""" prepare labels """
df.loc[(df['trade'] == 1)& (df.macd > df.macdsignal), 'position'] = 'bottom' 
df.loc[(df['trade'] == 1)& (df.macd < df.macdsignal), 'position'] = 'top'
# df.loc[(df['trade'] == 1)& (df.sma9 > df.sma21), 'position'] = 'top' 
# df.loc[(df['trade'] == 1)& (df.sma9 < df.sma21), 'position'] = 'bottom'
# df.loc[(df['position'] == 'top') & (df['label'] == 'long')]
# df.loc[(df['position'] == 'bottom') & (df['label'] == 'short')]


