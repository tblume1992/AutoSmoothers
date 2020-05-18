# AutoSmoothers
Class Arguments:
  Smoothers: 
    1.'brown'- doubley exponential smoothed, takes alpha.
    2.'simple'- simple exponential smoothing, takes alpha.
    3.'double'- double exponential smoothing, take alpha and beta.
    4.'dampened_double'- dampened double exponential smoothing, takes alpha, beta, damp (damp factor between 0-1)
    5.'brown_linear'- brown's double exponential smoothing where the level is smoothed twice, takes alpha, beta

  Smooth History:
    Smooth history before fitting, True will do simple exponential smoothing while 'brown' will do doubley smoothed.
    
  Clean Outliers:
    Cleans outliers with IQR, set the iqr_multiplier to change the tolerance where 3 is more leniant than 1.5.  Outliers are replace
    with the mean of the series excluding the outliers themselves.

  Class Methods:
    Fit: fits given smoother with required arguments. Returns fitted, predicted, and cleansed history.
    Sample: samples our objective function (holdout error).  Uses multiprocessing and assigns a chain per worker.  Must run in __main__ 
    for windows. Returns list of tuples from the sample as well as the list of fitted + predicted values.
    Optimize: optimizes our objective function in an attempt to find a stable solution (i.e. a cluster center achieved from running 
    'sample').  As such, it likely will not coincide with the minimum holdout error. Returns the optimal alpha/beta. 



Standard Fit:
```python
import quandl
import pandas as pd
import AutoSmoothers
if __name__ == '__main__':
    data = quandl.get("BITSTAMP/USD")
    y = data['High']
    y = pd.Series(y[-450:].values)
    smoother = AutoSmoothers.AutoSmoothers(series = y,
                             forecast_horizon = 50,
                             smoother = 'double',
                             alpha = .5,
                             beta = .5,
                             smooth_history = False,
                             clean_outliers = False
                             )
    fitted, predicted, series = smoother.fit()
```
Optimize to find optimal coefficients then refit with those. Guess I need to have it just return the optimal model instead to avoid 
refitting
```python
import quandl
import pandas as pd
import AutoSmoothers
if __name__ == '__main__':
    data = quandl.get("BITSTAMP/USD")
    y = data['High']
    y = pd.Series(y[-450:].values)
    smoother = AutoSmoothers.AutoSmoothers(series = y,
                             forecast_horizon = 50,
                             smoother = 'double',
                             smooth_history = False,
                             clean_outliers = False
                             )
    optimal_coefs = smoother.optimize()
    fitted, predicted, series = AutoSmoothers.AutoSmoothers(series = y,
                         forecast_horizon = 50,
                         smoother = 'double',
                         alpha = optimal_coefs[0],
                         beta = optimal_coefs[1],
                         smooth_history = False,
                         clean_outliers = False
                         ).fit()
```
Example of Sample, lots of parameters to consider such as tolerance for accepting a proposal (between 1-10) and burn_in.
```python
import quandl
import pandas as pd
import AutoSmoothers
if __name__ == '__main__':
    data = quandl.get("BITSTAMP/USD")
    y = data['High']
    y = pd.Series(y[-450:].values)
    smoother = AutoSmoothers.AutoSmoothers(series = y,
                             forecast_horizon = 50,
                             smoother = 'double',
                             smooth_history = False,
                             clean_outliers = False
                             )
    results, sampled = smoother.sample(iterations = 2000, max_draws = 20000, tolerance = 6, burn_in = 500)
    #plot the samples fitted + holdout predicted
    flat_list = [item for sublist in sampled for item in sublist]
    flat_list = [np.append(i[0], i[1]) for i in flat_list]
    for idx, i in enumerate(flat_list):
        plt.plot(i, alpha = .05, color = 'blue')
        
    avg_samples = np.median(flat_list, axis = 0)
    plt.plot(avg_samples, color = 'black')
    plt.plot(y, color = 'red')
    plt.show()
    #plot the sampled alpha, betas with holdout mse
    output= pd.DataFrame(results, columns = ['Alpha', 'Beta', 'MSE'])
    min_mse_index = output['MSE'].idxmin() 
    optimal_alpha = output['Alpha'][min_mse_index]
    optimal_beta = output['Beta'][min_mse_index]
    plt.scatter(output['Alpha'].values, output['Beta'].values, cmap=cm.jet, c=-output['MSE'].values, s=10)
    plt.axvline(x = output['Alpha'][min_mse_index], ymin = 0, ymax = output['Beta'][min_mse_index])
    plt.axhline(y = output['Beta'][min_mse_index], xmin = 0, xmax = output['Alpha'][min_mse_index])
    plt.plot(optimal_coefs[0], optimal_coefs[1], marker='x', markersize=10, color="black")
    plt.show()
    plt.hist(output['Alpha'], label = 'alpha')
    plt.show()
    plt.hist(output['Beta'], label = 'beta')
    plt.show()
```


