import numpy as np
import cupy as cp

def convert_arrays(X,y):
    X_cp = cp.array(X)
    y_cp = cp.array(y)
    return X_cp, y_cp

def naive_OLS(X,y):
    XTX_inv = cp.linalg.inv(cp.dot(X.T, X))
    XTy = cp.dot(X.T, y)
    beta = cp.dot(XTX_inv, XTy)
    return beta

def default_OLS(X,y):
    return cp.linalg.lstsq(X,y,rcond=None)[0]

def cholesky_inv(A):
    """
    Efficiently finds the inverse of A, a PSD, Hermitian matrix.
    """
    L = cp.linalg.cholesky(A)
    Y = cp.linalg.solve(L,cp.eye(A.shape[0]))
    return cp.linalg.solve(L.T,Y)

def spherical_error(X,y,beta):
    """
    Computes the spherical error for X and y.
    Spherical error assumes no serial correlation and that all errors have the same variance.
    """
    # Calculate error
    e = y - cp.dot(X,beta)
    # Calculate estimated variance of error
    sigma2 = cp.dot(e.T,e) / (X.shape[0] - X.shape[1])
    # Calculate spherical error
    se = cp.sqrt(sigma2 * cholesky_inv(cp.dot(X.T,X))).diagonal()
    return se

def robust_error(X,y,beta,method='HC0'):
    """
    Computes heteroskedastic robust errors for X and y.
    This assumes no serial correlation.
    By default HC0 is used. The caller is responsible for switching to HC1, HC2, or HC3 when those are most optimal.
    """
    # Calculate error
    e = y - cp.dot(X,beta)
    # Calculate estimated variance of error
    sigma2 = cp.dot(e.T,e) / (X.shape[0] - X.shape[1])
    if method == 'HC0':
        cinv = cholesky_inv(cp.dot(X.T,X))
        se = cp.sqrt(cinv @ X.T @ cp.diag(e**2) @ X @ cinv).diagonal()
    
    return se