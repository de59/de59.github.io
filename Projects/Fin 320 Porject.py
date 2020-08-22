import pandas as pd
import datetime
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
import seaborn as sns

def get_single_stock_data(start_date, end_date, symbol):
    data = web.DataReader(symbol, "yahoo", start_date, end_date)
    return data

start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2019, 1, 1)
symbol = "IBM"

data = lambda x : get_single_stock_data(start_date,end_date,x)
datad = lambda x,start_date,end_date : get_single_stock_data(start_date,end_date,x)
d1_rtrn = lambda x : x['Adj Close'].pct_change(1).dropna()
std = lambda x : np.std(x)

def correlation_over_time(prtf_tkrs, num_yrs):
    start = 2014
    end = 2015
    corr_yr = pd.DataFrame()
    for i in range(num_yrs):
        start_date = datetime.datetime(start, 1, 1)
        end_date = datetime.datetime(end, 1, 1)
        x = d1_rtrn(datad(prtf_tkrs[0], start_date, end_date))
        df = pd.DataFrame(x)
        for i in range(1, len(prtf_tkrs)):
            y = d1_rtrn(datad(prtf_tkrs[i], start_date, end_date))
            df1 = pd.DataFrame(y)
            df = pd.concat([df, df1['Adj Close']], axis=1)
        df.columns = prtf_tkrs
        corr = df.corr()
        corr.columns = ['SPY', start]
        corr_yr = pd.concat([corr_yr, corr[start]], axis=1, sort=True)
        start += 1
        end += 1
    return corr_yr

#print(correlation_over_time(['SPY', 'IAU'], 4).loc['SPY'])

def removing_outlier(prtf_tkrs):
    start = 2018
    end = 2019
    start_date = datetime.datetime(start, 1, 1)
    end_date = datetime.datetime(end, 1, 1)
    x = d1_rtrn(datad(prtf_tkrs[0], start_date, end_date))
    df = pd.DataFrame(x)
    for i in range(1, len(prtf_tkrs)):
        y = d1_rtrn(datad(prtf_tkrs[i], start_date, end_date))
        df1 = pd.DataFrame(y)
        df = pd.concat([df, df1['Adj Close']], axis=1)
    df.columns = prtf_tkrs
    return df

def outlier(df):
    outliers = pd.DataFrame()
    for i in range(len(df.columns)):
        df2 = pd.DataFrame(df.iloc[:, (i)])
        for x in range(251):
            rtn = df2.iloc[x-1]
            if np.abs(rtn[0]) > np.abs(mquantiles(df2, [0.05])):
                out = pd.DataFrame(df.iloc[x, :])
                outliers = pd.concat([outliers, out], axis=1)
        return outliers

def plt_heatmap(df):
    a = pd.DataFrame()
    for i in range(df.shape[0]):
        b = pd.DataFrame(df.iloc[i])
        a = pd.concat([a, b], axis=1)
    data = pd.DataFrame(a.corr())
    sns.heatmap(data, vmin=-1, vmax=1,
                xticklabels=data.columns,
                yticklabels=data.columns)
    return plt.show()

test = removing_outlier(['SPY', 'QQQ', 'DIA', 'IOO', 'VEA', 'EXI2.DE', 'EET', 'VWO', 'SCHE', 'GOVT', 'SHV', 'TLH', 'PICB', 'EMAG', 'HYMB', 'VTEB', 'ZROZ', 'TPX', 'RWO', 'SCHH', 'DBC', 'GCC', '^SP500TR'])#['SPY', 'SHV', 'IAU', 'QQQ', 'DIA', 'IOO', 'VEA'])
#['SPY', 'QQQ', 'DIA', 'IOO', 'VEA', 'EXI2.DE', 'EET', 'VWO', 'SCHE', 'GOVT', 'SHV', 'TLH', 'PICB', 'SPXB', 'EMAG', 'HYMB', 'VTEB', 'ZROZ', 'TPX', 'RWO', 'SCHH', 'DBC', 'GCC', '^SP500TR']
test1 = pd.DataFrame(outlier(test))

print(plt_heatmap(test1))

