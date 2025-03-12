
import numpy as np
from scipy.optimize import minimize

# Use multiprocessing to speed up the optimization
def worker(W0, feature, Q_train, N_train, method, param_bnd,c_weight = [1e-3,1e-2]):
    """Worker function to perform optimization."""
    alpha = 1e-6
    rho = 0.95
    cons = [
        {'type': 'ineq', 'fun': g1, 'args': (feature,)},
        {'type': 'ineq', 'fun': g2, 'args': (feature,)},
    ]
    res = minimize(
        end_to_end_loss,
        W0,
        args=(feature, N_train, Q_train, alpha, rho,c_weight),
        bounds=((param_bnd,) * len(W0)),
        constraints=cons,
        method=method,
        options={'disp': False, 'maxiter': 500}
    )
    return res

# Power-law terms, two cell specific parameters.
def end_to_end_loss(W_,X,N,Q,alpha,rho,c_weight = [1e-3,1e-2]):
    # Without constraining the absolute values and add scaling for each X*wi
    n_features = X.shape[1]
    W = W_.reshape(n_features,2) # First row (or the first two elements in the flatten input) is the intercept
    len_interp = Q.shape[1]
    I_nm = np.ones_like(Q)
    I_1m = np.ones((1,len_interp))

    w1 = W[:,0].reshape(-1,1)
    w2 = W[:,1].reshape(-1,1)

    # Local parameters
    c1 = X@w1*c_weight[0]
    c2 = X@w2*c_weight[1]

    # Terms inside the norm
    term_2 = c1@I_1m*N**(c2@I_1m)

    loss_1 = np.mean(np.nanmean(np.array(I_nm - term_2 - Q )**2,axis=1))
    loss_2 = np.sum(W[2:] ** 2) * 0.5 # L2 norm
    loss_3 = np.sum(np.abs(W[2:])) # L1 norm
    loss = loss_1 * 100 + alpha*(rho*loss_3 + (1-rho)*loss_2)
    return loss

# Define two nonlinear constraints to ensure c1 and c2 are all positive
def g1(W_,X_):
    n_features = X_.shape[1]
    W = W_.reshape(n_features,2)
    w1 = W[:,0].reshape(-1,1)    
    
    X = np.mat(X_)
    # Terms inside the norm
    c1 = X*w1
    return np.min(c1)

def g2(W_,X_):
    n_features = X_.shape[1]
    W = W_.reshape(n_features,2)
    X = np.mat(X_)
    w2 = W[:,1].reshape(-1,1)
    # Terms inside the norm
    c2 = X*w2
    return np.min(c2)