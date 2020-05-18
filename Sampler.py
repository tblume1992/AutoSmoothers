import numpy as np
import scipy.stats

def sample(kwargs, chain):
    chains = kwargs['chains']
    iterations = kwargs['iterations']
    max_draws = kwargs['max_draws']
    tolerance = kwargs['tolerance']
    burn_in = kwargs['burn_in']
    function = kwargs['function']
    series = kwargs['series'] 
    test_set = kwargs['test_set']  
    holdout = kwargs['holdout']
    init_steps = kwargs['init_steps']
    iterations = int(iterations/chains)
    iteration = 0
    burn_in_mse = []
    expon_stats = [0,0]
    output = []
    last_mse = 10**5
    sampled_predictions = []
    while len(output) != iterations + burn_in + init_steps:
      iteration += 1
      if len(output) >= init_steps:
          expon_stats = scipy.stats.expon.fit(burn_in_mse)
      if iteration == max_draws + init_steps + burn_in:
          break
      alpha = np.random.random()
      beta = np.random.random()
      while alpha < .01 or alpha > .99:
        alpha = np.random.random()
      while beta < .01 or beta > .99:
        beta = np.random.random()
      kwargs['alpha'] = alpha
      kwargs['beta'] = beta
      predictions, test_predictions = function(series = series, num_steps = holdout, kwargs = kwargs)
      mse = np.mean((test_set.reset_index(drop = True) - test_predictions.reset_index(drop = True))**2)
      if len(output) < init_steps:
        burn_in_mse.append(mse)
      if mse < last_mse \
             or scipy.stats.expon(expon_stats[0], expon_stats[1]).cdf(mse)/tolerance > tolerance*np.random.random() \
             or len(output) <= init_steps:
        output.append((alpha,beta,mse))
        sampled_predictions.append([predictions, test_predictions])
        last_mse = mse
    output = output[init_steps + burn_in:]
    sampled_predictions = sampled_predictions[init_steps + burn_in:]
    results = {'results': output,
               'samples': sampled_predictions}
    
    return results


def optimize(kwargs):
    function = kwargs['function']
    series = kwargs['series'] 
    test_set = kwargs['test_set'] 
    x1 = np.arange(0.1, .99, .1)
    init_points = []
    for i in x1:
        for j in x1:
            init_points.append((i, j))
    if kwargs['smoother'] in ['brown', 'simple']:
        init_points = list(set([i[0] for i in init_points]))
        optimal, weighted = search(list(zip(init_points, init_points)), 
                                   function, series, test_set, kwargs)
    else:
        optimal, weighted = search(init_points, function, series, test_set, kwargs)

    search_points = [
        (optimal[0], optimal[1]),
        (optimal[0] - .05, optimal[1]),
        (optimal[0] + .05, optimal[1]),
        (optimal[0] - .05, optimal[1] - .05),
        (optimal[0] + .05, optimal[1] + .05),
        (optimal[0], optimal[1] - .05),
        (optimal[0], optimal[1] + .05),
        (optimal[0] + .05, optimal[1] - .05),
        (optimal[0] - .05, optimal[1] + .05)
        ] 
    if kwargs['smoother'] in ['brown', 'simple']:
        search_points = list(set([i[0] for i in search_points]))
        optimal, weighted = search(list(zip(search_points, search_points)), 
                                   function, series, test_set, kwargs)
        
    else:
        optimal, weighted = search(search_points, function, series, test_set, kwargs)
    
    return weighted

def search(init_points, function, series, test_set, kwargs):   
    test_mse = []
    for i in init_points:
        kwargs['alpha'] = i[0]
        kwargs['beta'] = i[1]
        predictions, test_predictions = function(series = series, num_steps = len(test_set), kwargs = kwargs)
        test_mse.append(np.mean((test_set.reset_index(drop = True) - test_predictions)**2))
    
    from scipy import spatial
    init_points = np.asarray(init_points)
    tree = spatial.KDTree(init_points)
    avg_error = []
    var_error = []
    kernel = []
    for i in range(len(init_points)):
        pts = np.array([[init_points[i, 0], init_points[i, 1]]])
        if .1 in init_points[i] or .9 in init_points[i]:
            nn = tree.query(pts, k = 4)[1]
            kernel.append(nn)
            error_list = [test_mse[j] for j in range(len(test_mse)) if j in nn[0]]
            avg_error.append(np.mean(error_list)/100)
            var_error.append(np.sqrt(np.var(error_list)))
            
        else:
            nn = tree.query(pts, k = 5)[1]
            kernel.append(nn)
            error_list = [test_mse[j] for j in range(len(test_mse)) if j in nn[0]]
            avg_error.append(np.mean(error_list)/100)
            var_error.append(np.sqrt(np.var(error_list)))
    
    avg_test_error = np.mean(avg_error)
    avg_error = [i/avg_test_error for i in avg_error]
    var_test_error = np.mean(var_error)
    var_error = [i/var_test_error for i in var_error]
    
    optimal = [avg_error[i] + var_error[i] for i in range(len(var_error))]
    optimal_init_points = init_points[optimal.index(min(optimal))]
    weighted = np.average(init_points, weights = optimal, axis = 0)
    
    
    return optimal_init_points, weighted
