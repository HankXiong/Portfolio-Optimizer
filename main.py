# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:37:01 2021

@author: Hank Xiong
"""

import pandas as pd
import numpy as np
import os
from port_class import PortfolioOptimzer

files = os.listdir('./data')

df = pd.Series(dtype=np.float64)
for file in files:
    df_tmp = pd.read_csv(os.path.join('data',file),index_col=0,infer_datetime_format=True)['Adj Close']
    df_tmp = df_tmp.rename( file[:-4] )
    df = pd.concat([df,df_tmp],axis = 1,join = 'outer',sort = True)
    
df.drop(0,axis = 1,inplace=True)
df.index = pd.to_datetime(df.index)
df = df.interpolate(method = 'linear',axis = 0,limit_direction='forward')

## downsample to weekly price
df_week = df.resample(pd.offsets.Week(weekday=0),closed = 'left',label='left').bfill()

#df.isna().sum()
log_ret_df = np.log(df_week) - np.log(df_week).shift(1)
log_ret_df.dropna(axis = 0,inplace = True)

## empirical covariance
cov = log_ret_df.cov()
## expected return
exp_ret = log_ret_df.mean() * 52


bounds = (0,1) ## ((0,1),) * cov.shape[0]
uncertain_scale = 1.5



P = PortfolioOptimzer()

w = P.equal_weight(len(cov))
print('equal weight is', w)

w = P.inverse_vol(cov)
print('inverse volatility weight is', w)

w = P.max_diver_ratio(cov,penalty=0.1)
print( 'maximum diversification ratio weight is', w['x'])

w = P.target_return_robustMV(exp_ret,cov,0.05,shrink_size = 500,uncertain_scale = 1.5)
print('target return robust minimum variance weight is ', w['x'])

w = P.penalized_robustMV(exp_ret,cov,penalty=0.01, shrink_size=500,uncertain_scale=1.5)
print('penalized robust minimum variance weight is ', w['x'])

w = P.risk_parity(cov)
print('risk parity weight is', w['x'])

w = P.min_CVaR_loss(log_ret_df,alpha = 0.95)
print('minimum CVaR at 95% is', w['x'])
