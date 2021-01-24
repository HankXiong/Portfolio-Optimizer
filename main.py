# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:37:01 2021

@author: Lenovo
"""

import pandas as pd
import numpy as np
import os
#os.chdir(r'D:\Files\GitProjects\portfolio optimization')
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


cov = log_ret_df.cov()
log_ret_df.corr()


exp_ret = log_ret_df.mean()


import scipy.optimize as sco
import cvxpy as cp

bounds = (0,0.4)
uncertain_scale = 1.5

def return_target_mean_variance(exp_ret, cov, target_ret = 0, bounds = (0,1),shrink_size = 1000, uncertain_scale = 0 ):
    n = len(exp_ret)
    weight = cp.Variable(n)
    exp_ret = np.array(exp_ret)
    cov = np.array(cov)
    ret_stde_root = np.sqrt( np.diag(np.diag(cov))/ shrink_size )

    ## recycle if only give bound for one variable
    if not isinstance(bounds[0],tuple):
        lower = bounds[0]
        upper = bounds[1]
    else:
        lower = np.array([tmp[0] for tmp in bounds])
        upper = np.array([tmp[1] for tmp in bounds])
    
    cons = [exp_ret @ weight - uncertain_scale * cp.norm(weight @ ret_stde_root,2) >= target_ret,
                  np.ones(n) @ weight == 1,
                  weight >= lower,
                  weight <= upper
                  ]
   
    prob = cp.Problem(  cp.Minimize( cp.quad_form(weight, cov) ),
                      constraints = cons )
    prob.solve()
    return {'weight': weight.value, 'obj': prob.value}


def penalized_robust_mean_variance(exp_ret, cov, penalty = 0.1, bounds = (0,1),shrink_size = 1000, uncertain_scale = 0  ):
    n = len(exp_ret)
    weight = cp.Variable(n)
    exp_ret = np.array(exp_ret)
    cov = np.array(cov)
    ret_stde_root = np.sqrt( np.diag(np.diag(cov))/ shrink_size )
    ## recycle if only give bound for one variable
    if not isinstance(bounds[0],tuple):
        lower = bounds[0]
        upper = bounds[1]
    else:
        lower = np.array([tmp[0] for tmp in bounds])
        upper = np.array([tmp[1] for tmp in bounds])
        
    cons = [ np.ones(n) @ weight == 1,
                  weight >= lower,
                  weight <= upper
                  ]

    prob = cp.Problem(  
        cp.Minimize( 
            cp.quad_form(weight, cov) - penalty * (exp_ret @ weight) + uncertain_scale * cp.norm(weight @ ret_stde_root,2)
            ),
            constraints = cons 
        )

    prob.solve()
    return {'weight': weight.value, 'obj': prob.value}
    

def risk_parity(cov, obj_formula_type = 1):
    n = len(cov)
    if obj_formula_type == 1:
        ## formula 1
        weight = cp.Variable(n,nonneg=True)
        cons = [weight >= 0]
        prob = cp.Problem( 
            cp.Minimize( 
                0.5 * cp.quad_form(weight,cov) - np.ones(n) @ cp.atoms.elementwise.log.log(weight)
                ),
            constraints = cons
            )
        prob.solve()
        print('optimization is ' + prob.status)
        res = {'weight': weight.value/weight.value.sum()}
    else:
        ## formula 2  ## cp.sum_squares(cp.diag(weight) @ (cov @ weight) - aux_var)
        def riskparity_obj(x):
            ## the last variable is an auxiliary variable which will be the risk contribution of each component
            w=np.mat(x[:-1]).T
            ## enlarge the covariance to make convergence more accurate
            obj = np.square(np.diag(x[:-1]) * (cov*10000 * w) - x[-1]).sum()
            return obj
        cons = ({'type': 'eq', 'fun': lambda w:  sum(w[:-1]) -1})
        bnds = ((0, 1),) * n + ((0,None),)
        w_ini = np.ones(n) / n
        w_ini = np.insert(w_ini,n,1)
        res = sco.minimize(riskparity_obj, w_ini, bounds=bnds,constraints=cons,options={'disp':True,'ftol':10**-10})
        res = {'weight': res['x']}
    return res 
    

def MaxDiverwgtfind(sigma,penalty = 0., bounds = (0,1) ):
    n = len(sigma)
    sigma_pena = sigma + np.eye(n) * penalty
    diversification=lambda x: -np.dot( np.sqrt(np.diag(sigma_pena)),x).sum()/np.sqrt(np.dot(x.T,np.dot(sigma_pena,x)))
    cons=({'type':'eq', 'fun':lambda x: np.nansum(x)-1})
    w_ini=np.ones(n) / n
    bnds= ((0,1),)*7
    result=sco.minimize(diversification,w_ini,bounds=bnds,constraints=cons,options={'ftol':10**-8,'disp':True})

