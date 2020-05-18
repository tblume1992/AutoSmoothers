import pandas as pd
import numpy as np
import scipy.stats
import sampler
import multiprocessing as mp
from functools import partial

class AutoSmoothers:
    def __init__(self, series, 
                 forecast_horizon = 50, 
                 smoother = 'simple', 
                 clean_outliers = False, 
                 iqr_multiplier = 1.5,
                 smooth_history = False, 
                 **kwargs):
        
        self.series = series.copy()     
        if smooth_history:
            self.get_smoothed_history(smooth_history)                    
        self.clean_outliers = clean_outliers
        self.forecast_horizon = forecast_horizon
        self.iqr_multiplier = iqr_multiplier
        self.kwargs = kwargs      
        self.set_model(smoother) 
        self.smoother = smoother
        
        return 

    def set_model(self, smoother):
        if smoother == 'double':
            self.model = self.double_exponential_smoothing
            
        elif smoother == 'simple':
            self.model = self.simple_exponential_smoothing
            
        elif smoother == 'brown':
            self.model = self.brown_exponential_smoothing
            
        elif smoother == 'brown_linear':
            self.model = self.browns_linear_exponential_smoothing
        
        elif smoother == 'dampened_double':
            self.model = self.trend_dampened_double_exponential_smoothing
        
        return
            
    def get_smoothed_history(self, smooth_history):
        kwargs = {
            'alpha': .5,
            }
        if smooth_history == 'simple':
            self.series = self.simple_exponential_smoothing(series = self.series, num_steps = 0, kwargs = kwargs)[0]
        elif smooth_history == 'brown':
            self.series = self.brown_exponential_smoothing(series = self.series,num_steps = 0, kwargs = kwargs)[0]
        else:
            print('Select a valid smoother: "brown" or "simple". Defaulting to "simple"')
            self.series = self.simple_exponential_smoothing(series = self.series, num_steps = 0, kwargs = kwargs)[0]
        
        return
            
        
    def double_exponential_smoothing(self, series, num_steps, kwargs):
        alpha = kwargs['alpha']
        beta = kwargs['beta']
        result = [series[0]]
        for n in range(1, len(series)+num_steps):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): 
                value = result[-1]
            else:
                value = series[n]
            last_level, level = level, alpha*value + (1-alpha)*(level+trend)
            trend = beta*(level-last_level) + (1-beta)*trend
            result.append(level+trend)
            
        return pd.Series(result[:len(series)]), pd.Series(result[len(series):])
    
    def simple_exponential_smoothing(self, series, num_steps, kwargs):
        alpha = kwargs['alpha']
        if 'recurse' in kwargs.keys():
            recurse = kwargs['recurse']
        else:
            recurse = False
        result = [series[0]]
        for n in range(1, len(series)+num_steps):
            if n == 1:
                level = series.iloc[0]
            if n >= len(series): 
                value = result[-1]
            else:
                value = series.iloc[n]
            level = alpha*value + (1-alpha)*(level)
            result.append(level)
        if recurse:
            kwargs['recurse'] = False
            b_es = self.simple_exponential_smoothing(series = pd.Series(result), 
                                                num_steps = num_steps,
                                                kwargs = kwargs
                                                )[0]           
        else:
            b_es = result
        
            
        return pd.Series(b_es[:len(series)]), pd.Series(b_es[len(series):])
    
    def brown_exponential_smoothing(self, series, num_steps, kwargs):
        fitted, predicted = self.simple_exponential_smoothing(series, 
                                               num_steps, 
                                               kwargs
                                               )
        fitted, predicted = self.simple_exponential_smoothing(fitted, 
                                               num_steps, 
                                               kwargs
                                               )

            
        return fitted, predicted
    
    def browns_linear_exponential_smoothing(self, series, num_steps, kwargs):
        result = [series[0]]
        alpha = kwargs['alpha']
        beta =  kwargs['beta']
        for n in range(1, len(series)+num_steps):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): 
                value = result[-1]
            else:
                value = series[n]
            last_level, level = level, alpha*value + (1-alpha)*(level+trend)
            last_level, level = level, alpha*value + (1-alpha)*(level+trend)
            trend = beta*(level-last_level) + (1-beta)*trend
            result.append(level+trend)
            
        return pd.Series(result[:len(series)]), pd.Series(result[len(series):])
    
    def trend_dampened_double_exponential_smoothing(self, series, num_steps, kwargs):
        alpha = kwargs['alpha']
        beta = kwargs['beta']
        damp = kwargs['damp']
        result = [series[0]]
        for n in range(1, len(series)+num_steps):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): 
                value = result[-1]
            else:
                value = series[n]
            
            last_level, level = level, alpha*value + (1-alpha)*(level+damp*trend)
            trend = beta*(level-last_level) + (1-beta)*trend*damp
            if n <= len(series):
                result.append(level+trend)
            else:
                phi = [damp**(i) for i in range(n -len(y))]
                result.append(level+phi[-1]*trend)
            
        return pd.Series(result[:len(series)]), pd.Series(result[len(series):])
    
    def fit(self):
        if self.clean_outliers:
            q75, q25 = np.percentile(self.series, [75 ,25])
            iqr = q75 - q25
            lowerbound = q25 - self.iqr_multiplier * iqr
            upperbound = q75 + self.iqr_multiplier * iqr
            mean = np.mean(self.series[(self.series > lowerbound) & (self.series < upperbound)])
            self.series[self.series > upperbound] = mean
            self.series[self.series < lowerbound] = mean
        fitted, predicted = self.model(series = self.series, 
                                       num_steps = self.forecast_horizon, 
                                       kwargs = self.kwargs)
        
        return fitted, predicted, self.series
        
    def sample(self, 
                iterations = 1000,
                max_draws = 5000, 
                burn_in = 500, 
                init_steps = 500,
                tolerance = 10, 
                chains = 4, 
                holdout = 50):
        if tolerance > 10:
            tolerance = 10
        if tolerance < 1:
            tolerance = 1
        tolerance = 11 - tolerance
        kwargs = {
        'function': self.model,
        'series': self.series[:-holdout],
        'test_set': self.series[-holdout:],
        'chains': chains,
        'iterations': iterations,
        'max_draws': max_draws,
        'tolerance': tolerance,
        'burn_in': burn_in,
        'holdout': holdout,
        'init_steps': init_steps,
        'smoother': self.smoother
        }
        
        with mp.Pool(chains) as pool:
            results = list(pool.map(partial(sampler.sample, kwargs), list(range(chains))))
        sampled = [i['samples'] for i in results ] 
        results = [i['results'] for i in results]        
        
        for i, result in enumerate(results):
            if len(result) < int(iterations/chains):
                print(f'Chain {i} failed to converge. Try increasing tolerance')
        results = [item for sublist in results for item in sublist]
        
        return results, sampled
    
    def optimize(self, holdout = 50):
        kwargs = {
        'function': self.model,
        'series': self.series[:-holdout],
        'test_set': self.series[-holdout:],
        'smoother': self.smoother
        }
        
        results = sampler.optimize(kwargs)
        
        return results
