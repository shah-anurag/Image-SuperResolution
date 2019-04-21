import numpy as np
EPSILON = 1e-3
def rho(x):
    return np.sqrt(x**2, EPSILON**2)
def gradient_difference(result, target):
    diff = np.subtract(result, target)
    
def loss_rho(result, target):
    diff = np.subtract(result, target)
    sqre = np.square(diff)
    sqre = np.add(sqre, EPSILON**2)
    rho = np.sqrt(sqre)
    rho = np.sum(rho)
    return rho
    
