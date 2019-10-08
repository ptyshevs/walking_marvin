import numpy as np

def sample_like(weights, sigma=1, rs=None):
    """
    Create a sample of the same shapes as the input
    @param weights: list of np.arrays
    """
    
    if rs is None:
        rs = np.random
    
    return [rs.randn(*l.shape) * sigma for l in weights]


def combine_weights(params, delta_params, sigma):
    return [W + dW * sigma for W, dW in zip(params, delta_params)]


def update_params(params, population, rewards, lr=0.05, sigma=0.1):
    """
    Inplace update of parameters
    """
    n = len(population)
    for i in range(len(params)):
        W = params[i]
        
        dW_accum = np.zeros_like(W)
        for candidate, reward in zip(population, rewards):
            dW = candidate[i]
            dW_accum += reward * dW
        W_new = W + lr / (n * sigma) * dW_accum
        params[i] = W_new
    return params