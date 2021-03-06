import numpy as np
from scipy.special import logsumexp,softmax
from scipy import optimize
from sklearn.base import BaseEstimator

## huber log-likelihood and gradients

def huber_loss_general(x,scale):
  return scale**2 * ((1+(x/scale)**2)**0.5-1)

def huber_logp(scale):
  def huber_logp(x):
    return -huber_loss_general(x,scale)
  return huber_logp

def huber_logp_jac(scale):
  def huber_logp_jac(x):
    return -scale**2 * x/(1+(x/scale)**2)**0.5
  return huber_logp_jac

## mixture log-likelihood calculations 

def x_c_logp(x,c,logp_func=huber_logp(0.0)):
  ''' log-prob of data given cluster location '''
  return logp_func(x-c).sum(axis=1) # (n_rows,n_class)

def x_logp(x,c,y_logp,logp_func):
  ''' log-prob of data over clusters assignments and cluster centers '''
  logp = x_c_logp(np.expand_dims(x,-1),c,logp_func) + y_logp # (n_rows,n_class)
  return logsumexp(logp,axis=-1) # (n_rows,)

class HuberMixture(BaseEstimator):
  '''
    Represents the data as a mixture of components with Huber-like loglikelihoods.
    May perform much better than gaussian mixtures on certain datasets.
  '''

  def __init__(self,n_clusters=3,huber_scale=0.25):
    self.scale = huber_scale
    self.n_classes = n_clusters


  def _mstep(self,x,c,y_logp):
    ''' improves cluster locations given data and class log-probs (by numerical optimization) '''
    
    def loss_func(x,y_logp,c_dense_shape,logp_func):
      ''' note: needs to convert tensors between model (rank 2) and optimization subroutine (rank 1) '''
      def func(c):
        c = c.reshape(c_dense_shape)
        return -x_logp(x,c.reshape(c_dense_shape),y_logp,logp_func).mean()
      return func
    
    def loss_jac(x,y_logp,c_dense_shape,logp_func,logp_jac):
      ''' note: needs to convert tensors between model (rank 2) and optimization subroutine (rank 1) '''
      def jac(c):
        c = c.reshape(c_dense_shape)
        logp = x_c_logp(np.expand_dims(x,-1),c,logp_func) + y_logp # (n_rows,n_class)
        p = softmax(logp,-1) # (n_rows,n_class)
        jac = np.expand_dims(p,1) * -logp_jac(c-np.expand_dims(x,-1)) # (n_rows,n_features,n_class)
        jac = jac.mean(0) # (n_features,n_class)
        return jac.ravel()
      return jac

    ## optimize clusters
    c_dense_shape = c.shape
    logp_func = huber_logp(self.scale)
    logp_jac = huber_logp_jac(self.scale)
    out = optimize.minimize(loss_func(x,y_logp,c_dense_shape,logp_func),c.ravel(),jac=loss_jac(x,y_logp,c_dense_shape,logp_func,logp_jac))
    c = out.x.reshape(c_dense_shape)
    return c,-out.fun

  def _estep(self,x,c,y_logp):
    ''' updates class log-probs given cluster location (by bayes probability rules) '''
    logp_func = huber_logp(self.scale)
    logp = x_c_logp(np.expand_dims(x,-1),c,logp_func) + y_logp # (n_rows,n_class)
    return logp - np.expand_dims(logsumexp(logp,axis=1),1)

  def fit_predict(self,x,n_iter=30,n_init=3,warm_start=True):
    ## dispatch dimensions
    n_rows,n_features = x.shape
    n_classes = self.n_classes
    self.logp = -np.inf
    c = None
    y_logp = None
    for _ in range(n_init):
      ## warm start or initialize variables
      if c is None or not warm_start:
        c = np.random.normal(size=(n_features,n_classes))  # (n_features,n_class)
        y_logp = np.log(np.ones((n_rows,n_classes))/n_classes)
      else:
        c,y_logp = self.c,self.y_logp
      ## train by alternating M and E step
      for _ in range(n_iter):
        c,logp = self._mstep(x,c,y_logp)
        y_logp = self._estep(x,c,y_logp)
      if logp > self.logp + 1e-3:
        self.logp = logp
        self.c = c
        self.y_logp = y_logp
    return self.y_logp