# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:02:09 2021

@author: Hank Xiong
"""

import numpy as np
import scipy.optimize as sco
import cvxpy as cp

class PortfolioOptimzer():
    def __init__(self):
        ## blank constructor
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
        print('-----equal_weight-----')
        print('-----inver_vol-----')
        print('-----target_return_robustMV-----')
        print('-----penalized_robustMV-----')
        print('-----max_diver_ratio-----')
        print('-----risk_parity-----')
        print('-----min_CVaR_loss-----')
    
    def equal_weight(self,n):
        '''
        Parameters
        ----------
        n : Int
            number of components in the portfolio

        Returns
        -------
        numpy.array
        '''
        return np.ones(n) / n
    
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
        dict 
            {'x': weight}

        '''
        cov = np.array(cov)
        if vol_or_var == 'vol':
            inv_vols = 1 / np.sqrt(np.diag(cov))
        else:
            ## inverse variance
            inv_vols = 1 / np.diag(cov)
        weight = inv_vols / inv_vols.sum()
        
        return {'x': weight}
    
    def target_return_robustMV(self, exp_ret, cov, target_ret = 0, bounds = (0,1),shrink_size = 1000, uncertain_scale = 0 ):
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
        print('optimization reached ', prob.status)
        return {'x': weight.value, 'obj': prob.value}
    
    def penalized_robustMV(self, exp_ret, cov, penalty = 0.1, bounds = (0,1),shrink_size = 1000, uncertain_scale = 0):
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
            this number is used to estimate the standard error of return estimates by dividing covariance with this number
            The default is 1000.
        uncertain_scale : float, optional
            uncertain level bounding the extimated return, the higher the less sensitive of optimized return with regard to the extimated return
            The default is 0, usually can be quantile of normal distribution(95%,99% etc.)

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
        print('optimization reached ', prob.status)
        return {'x': weight.value, 'obj': prob.value}
    
    def risk_parity(self, cov, bounds = (0,1) ):
        '''
        equalize risk contribution of each component in the portfolio
        the risk contribution of component i is R_i = w_i * (cov * w)_i
        
        note that: long only to have the unique solution
        if bounds takes effect, then the risk contribution will not be equal
        
        Parameters
        ----------
        cov : numpy.array
            covariance matrix 
        
        bounds: tuple or nested tuple
            weight bound of each component    
            The default is (0,1).
        
        Returns
        -------
        dict
            {'x': weight, 'target_contribution': risk contribution}

        '''
        cov = np.array(cov)
        n = len(cov)
        '''
        lower,upper = self.set_bounds(bounds)
        ## formula 1
        weight = cp.Variable(n,nonneg=True)
        cons = [weight >= 0]
        prob = cp.Problem( 
            cp.Minimize( 
                0.5 * cp.quad_form(weight,cov) - np.ones(n) @ cp.log(weight)
                ),
            constraints = cons
            )
        prob.solve()
        print('optimization is ' + prob.status)
        res = {'x': weight.value/weight.value.sum()}
        '''
        ## formula 2  ## cp.sum_squares(cp.diag(weight) @ (cov @ weight) - aux_var)
        def riskparity_obj(x):
            ## the last variable is an auxiliary variable which will be the risk contribution of each component
            w=np.mat(x[:-1]).T
            ## enlarge the covariance to make convergence more accurate
            obj = np.square(np.diag(x[:-1]) * (cov*10000 * w) - x[-1]).sum()
            return obj
        cons = ({'type': 'eq', 'fun': lambda w:  sum(w[:-1]) -1})
        if not isinstance(bounds[0],tuple):
            bnds = (bounds,) * n + ((0,None),)
        else:
            bnds = bounds + ((0,None),)
        w_ini = np.ones(n) / n
        w_ini = np.insert(w_ini,n,1)
        prob = sco.minimize(riskparity_obj, w_ini, bounds=bnds,constraints=cons,options={'disp':True,'ftol':10**-8})
        print(prob['message'])
        res = {'x': prob['x'][:-1], 'target_contribution': prob['x'][-1]}
        
        return res
    
    def max_diver_ratio(self, cov, penalty = 0., bounds = (0,1) ):
        '''
        maximize diversification ratio of the portfolio (the proportion of accounted by its own variance is the largest)
        sum of component variance is sum(w * diag(cov))
        vol = sqrt(w^T * cov * w)
        diver_ratio = sum(w * diag(cov)) / vol
        
        Parameters
        ----------
        cov : numpy.array
            covariance matrix
        penalty : float, optional
            the penalty added on the diagonal variance to make portfolio less concentraded
            The default is 0.., typical value range from 0.01 to 1
        bounds : tuple or nested tuple
            weight bound of each component  
            The default is (0,1).

        Returns
        -------
        dict
            {'x':weight, 'diversification ratio': diversifcation ratio}

        '''
        
        cov = np.array(cov)
        n = cov.shape[0]
        ## add penalty to make investment less concentrated
        sigma_pena = cov * 1000 + np.eye(n) * penalty
        diver_ratio = lambda x: -np.dot( np.sqrt(np.diag(sigma_pena)),x).sum()/np.sqrt(np.dot(x.T,np.dot(sigma_pena,x)))
        cons=({'type':'eq', 'fun':lambda x: np.nansum(x)-1})
        w_ini=np.ones(n) / n
        
        ## define bounds for each component
        if not isinstance(bounds[0],tuple):
            bnds = (bounds,) * n 
        else:
            bnds = bounds

        prob = sco.minimize( diver_ratio,w_ini,
                            bounds=bnds,
                            constraints=cons,
                            options={'ftol':10**-8,'disp':True}
                            )
        print(prob['message'])
        
        return {'x': prob['x'], 'diversification ratio': -prob['fun']}
    
    def min_CVaR_loss(self,ret_hist, alpha = 0.9, bounds = (0,1) ):
        '''
        Minimize CVaR at alpha percentile given all return scenarios,
        this can be simplied into a linear programming problem by including auxiliary variables

        Parameters
        ----------
        ret_hist : numpy.array
            all possible return scenarios of portfolio component
        alpha : float, optional
            The confidence level at which CVaR will be optimized
            The default is 0.9.
        bounds : tuple or nested tuple, optional
            lower and upper bound for the portfolio component
            The default is (0,1).

        Returns
        -------
        DICT
            {'x' : component weight,
             'VaR': Value-at-Risk in alpha level,
             'CVaR': the optimized CVaR,
             'auxiliary variables': auxiliary variables}

        '''
        loss_hist = - np.array(ret_hist) ## convert to loss 
        n = loss_hist.shape[1]
        number_scenarios = loss_hist.shape[0]
        
        gamma = cp.Variable(1) ## will be VaR_alpha in optimization
        weight = cp.Variable(n)
        aux_z = cp.Variable(number_scenarios,nonneg=True)
        
        lower,upper = self.set_bounds(bounds)
        
        obj = gamma + 1 / (1 - alpha) / number_scenarios * cp.norm(aux_z,1)
        cons = [
                aux_z >= 0,
                aux_z >= loss_hist @ weight - gamma,
                np.ones(n) @ weight == 1,
                weight >= lower,
                weight <= upper
                ]
        prob = cp.Problem( 
            cp.Minimize(obj), constraints = cons
            )
        prob.solve()
        print('optimization reached ', prob.status)
        res = {'x': weight.value, 
               'VaR': gamma.value[0],
               'CVaR': prob.value,
               'auxiliary variables': aux_z.value}
        return res
    
    def set_bounds(self, bounds):
        ## recycle if only give bound for one variable
        if not isinstance(bounds[0],tuple):
                lower = bounds[0]
                upper = bounds[1]
        else:
            lower = np.array([tmp[0] for tmp in bounds])
            upper = np.array([tmp[1] for tmp in bounds])
        return  lower, upper
    