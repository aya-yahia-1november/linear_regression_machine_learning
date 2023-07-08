import numpy as np
def fit_module(x,betas):
    n=len(x)
    x_betas=np.ones((n,1))
    x=np.c_[x_betas,x]
    y_hat=x.dot(betas)
    return y_hat
def square_error(y,y_hat):
    error = y - y_hat
    square_error=error.dot(error.T)
    return square_error

def r_square(ssr,sst):
    r_square=ssr/sst
    return r_square


def find_betas(x, y):
  n=len(x)
  x_bias = np.ones((n, 1))
  x = np.c_[x_bias, x]
  betas = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
  return betas
def predict(x, betas):
    n = len(x)
    x_bias = np.ones((n, 1))
    x = np.c_[x_bias, x]
    prediction = x.dot(betas)
    return prediction
