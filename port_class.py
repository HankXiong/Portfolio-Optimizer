# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:02:09 2021

@author: Lenovo
"""

import pandas as pd
import numpy as np
import sklearn.covariance as skc

import cvxopt as cvx
from cvxopt import matrix
import cvxpy as cp

class PortfolioOptimzer():
    def __init__(self):
        '''
        self.exp_ret = None            
        self.cov = None       
        self.penalty = None
        self.lower = None
        self.upper = None
        self.target_return = None
        '''
    def __str__(self):
        return 'Portfolio optimizer object'
    
    def show_methods(self):
        print('-----target_return_RobustMV-----')
        print('put the target return as a strict constraint to let expected return satisfies with uncertainty allowance, then minimize the variance')
        print('-----penalized_RobustMV-----')
        print('set the expected return as a penalty term in the objective with uncertainty allowance, then do the minimization')
        print('MaxDiver')
        print('MinCVaR')
        print('EqualWeight')
        print('InverseVol')
        print('RiskParity')
    
    
    
    def target_return_RobustMV(self, exp_ret, cov, target_ret = 0, bounds = (0,1),shrink_size = 1000, uncertain_scale = 0 ):
        '''
        put the target return as a strict constraint to let expected return satisfies with uncertainty allowance, then minimize the variance
        Min: x^T * cov *  x
        s.t. 1^T x = 1;
             x >= lower bound
             x <= upper bound
             exp_ret ^T * x - uncertain_scale * expected return standard error >= target return
        

        Parameters
        ----------
        exp_ret : numpy.array
            expected return
        cov : numpy.array
            covariance matrix
        target_ret : numpy.array, optional
            the portfolio required expected return
            The default is 0.
        bounds :  tuple , optional
            (lower bound, upper bound) or specify the bound for each varianble
            The default is (0,1).
        shrink_size : Int, optional
            typically the sample size used to estimate the covariance matrix and expected return
            The default is 1000.
        uncertain_scale : float, optional
            uncertain level bounding the extimated return, the higher the less sensitive of optimized return with regard to the extimated return
            The default is 0.

        Returns
        -------
        dict
            {'x': weights, 'obj': the value of objective function}

        '''
        lower, upper = self.set_bounds(bounds)
        cov = np.array(cov)
        exp_ret = np.array(exp_ret)
        target_ret = np.array(target_ret)
        ret_stde_root = np.sqrt( np.diag(np.diag(cov))/ shrink_size )

        n = len(cov)
        
        weight = cp.Variable(n)
        cons = [
                 exp_ret @ weight - uncertain_scale * cp.norm(weight @ ret_stde_root,2) >= target_ret,
                  np.ones(n) @ weight == 1,
                  weight >= lower,
                  weight <= upper
                  ]
   
        prob = cp.Problem(  cp.Minimize(cp.quad_form(weight, cov)),
                      constraints = cons )
        prob.solve(reltol=1e-6)
        print(prob.status)
        return {'x': weight.value, 'obj': prob.value}
    
    def penalized_RobustMV(self, exp_ret, cov, penalty = 0.1, bounds = (0,1),shrink_size = 1000, uncertain_scale = 0):
        '''
        treat expected return as a penalty in the objective to do a risk return trade-off, also includes the uncertainty of the expected return, 
        then minimize objective
        
        Min: x^T * cov *  x - penalty * exp_ret ^T * x + uncertain_level * expected return standard error
        s.t. 1^T x = 1;
             x >= lower bound
             x <= upper bound
             

        Parameters
        ----------
        exp_ret : numpy.array
            expected return
        cov : numpy.array
           covariance matrix
        penalty : Int, optional
            The scale effect of expected return on the objective function
            The default is 0.1.
        bounds :  tuple , optional
            (lower bound, upper bound) or specify the bound for each varianble
            The default is (0,1).
        shrink_size : Int, optional
            typically the sample size used to estimate the covariance matrix and expected return
            The default is 1000.
        uncertain_scale : float, optional
            uncertain level bounding the extimated return, the higher the less sensitive of optimized return with regard to the extimated return
            The default is 0.

        Returns
        -------
        dict
            {'x': weights, 'obj': the value of objective function}

        '''
        lower, upper = self.set_bounds(bounds)
        cov = np.array(cov)
        exp_ret = np.array(exp_ret)
        ret_stde_root = np.sqrt( np.diag(np.diag(cov))/ shrink_size )

        n = len(cov)
        
        weight = cp.Variable(n)
        cons = [ np.ones(n) @ weight == 1,
                 weight >= lower,
                 weight <= upper
                 ]
        prob = cp.Problem(  cp.Minimize( cp.quad_form(weight, cov) - penalty * (exp_ret @ weight) + uncertain_scale * cp.norm(weight @ ret_stde_root,2) ),
                     constraints = cons )
        prob.solve(reltol=1e-6)
        print(prob.status)
        return {'x': weight.value, 'obj': prob.value}
    
    
    def inverse_vol(self,cov, vol_or_var = 'vol'):
        '''
        The weight of each component is inversely proportional to its volatility or variance

        Parameters
        ----------
        cov : numpy.array
            covariance matrix
        vol_or_var : str, optional
            if 'vol' then inversely proportion to the volatility, otherwise variance
            The default is 'vol'.

        Returns
        -------
        weight : numpy.array
            weight of the portfolio

        '''
        if vol_or_var == 'vol':
            inv_vols = 1 / np.sqrt(np.diag(cov))
        else:
            ## inverse variance
            inv_vols = 1 / np.diag(cov)
        weight = inv_vols / inv_vols.sum()
        
        return weight
    
    def set_bounds(self, bounds):
        ## recycle if only give bound for one variable
        if not isinstance(bounds[0],tuple):
                lower = bounds[0]
                upper = bounds[1]
        else:
            lower = np.array([tmp[0] for tmp in bounds])
            upper = np.array([tmp[1] for tmp in bounds])
        return  lower, upper
    