import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf

class data_generator():
    """Abstract class for input-output time series generators of an ODE system
    """

    def __init__(self,T,n_t,delta_t,n_samples,n_dim):
        """Initializer

        :param T: the maximal t span 
        :type T: int, optional
        :param n_t: number of sample data for each trajectory
        :type n_t: int
        :param delta_t: time step for Euler's method 
        :type delta_t: float
        :param n_samples: number of trajectories
        :type n_ssamples: int
        :param n_dim: number of input features
        :type n_dim: int 

        """
        self.T = T
        self.n_t = n_t
        self.n_samples = n_samples
        self.delta_t = delta_t
        self.n_dim = n_dim
    
    def generate_initial_conditions(self, region):
        """Generate initial conditions for ODE

        :param region: the region of the initial conditions
        :rtype region: ndarray of shape [n_dim * 2,]
        :return: tensors of initial conditions of shape
                [n_samples, n_dim]
        :rtype: ndarray
        """
        initial = np.zeros((self.n_samples,self.n_dim))
        for i in range(self.n_dim):
            initial[:,i] = np.random.uniform(region[i*2],region[i*2+1],self.n_samples)
        return initial
    
    def generate_data(self,region,fun):
        """Generate data
        Generate an input-output time series dataset (X,Y) of the ODE system dh(u)/dt = f(t,u)
        X is evaluated at iT/n_t and y is evaluated at iT/n_t + delta_t


        :param region: the region of the initial conditions
        :rtype region: ndarray of shape [n_dim * 2,]
        :param fun: the right hand side of an ODE system dh(u)/dt = f(t,u)
        :rtype fun: function
        :return: (inputs, outputs) each of shape [n_samples,n_t,n_dim]
        :rtype: tuple
        """
        t_grid = np.linspace(0,self.T,self.n_t+1)
        t_grid_x = t_grid[0:-1]
        t_grid_y = t_grid_x + self.delta_t
        t_span = (t_grid[0], t_grid[-1])
        initial = self.generate_initial_conditions(region)
        X_data = np.zeros((self.n_samples,self.n_t,self.n_dim))
        Y_data = np.zeros((self.n_samples,self.n_t,self.n_dim))
        for i in range(self.n_samples):
            sol = solve_ivp(
                fun = fun,
                t_span = t_span,
                y0 = initial[i,:],
                method = 'RK45',
                rtol=1.e-8,
                atol=1.e-8,
                dense_output=True
            )
            X_data[i,:,:] = sol.sol(t_grid_x).T
            Y_data[i,:,:] = sol.sol(t_grid_y).T
        return X_data,Y_data
    
    def train_test_split(self,X_data,Y_data,train_size = 0.6, valid_size = 0.2):
        """split the train, valid, test dataset from X_data and Y_data

        :param X_data: time series of input variables
        :rtype X_data: ndarray of shape [n_samples,n_t,n_dim]
        :param Y_data: time series of putput variables
        :rtype Y_data: ndarray of shape [n_samples,n_t,n_dim]
        :param train_size: the proportion of train data 
        :type: float 
        :param valid_size: the proportion of valida data
        :type: float
        :return: (X_train,X_valid,X_test,Y_train,Y_valid,Y_test), 
                 the train, valid, test data of inputs and outputs
        :rtype: tuple
        """
        n = self.n_t
        X_train = X_data[:,0:int(n*train_size),:].reshape(-1,self.n_dim)
        X_valid = X_data[:,int(n*train_size):
                         int(n*(train_size + valid_size)),:].reshape(-1,self.n_dim)
        X_test = X_data[:,int(n*(train_size+valid_size)):,:].reshape(-1,self.n_dim)
        Y_train = Y_data[:,0:int(n*train_size),:].reshape(-1,self.n_dim)
        Y_valid = Y_data[:,int(n*train_size):
                         int(n*(train_size + valid_size)),:].reshape(-1,self.n_dim)
        Y_test = Y_data[:,int(n*(train_size+valid_size)):,:].reshape(-1,self.n_dim)

        return X_train,X_valid,X_test,Y_train,Y_valid,Y_test
    
    
    

class data_generator_sde():
    """Abstract class for input-output time series generators of an SDE system
    """
    def __init__(self,drift,diffusitivity):
        """Initializer

        for an SDE dX_t = f(X_t)dt + \sigma(X_t)dW_t
        :param drift: the frift term f(X_t)
        :type drift: function
        :param diffusitivity: the diffusion term \sigma(X_t)
        :type diffusitivity: function
        """
        self.drift = drift
        self.diffusitivity = diffusitivity
        
    def euler_sde(self,x_data,h):
        n_samples = x_data.shape[0]
        n_dim = x_data.shape[1]
        h = tf.cast(h,tf.float64)
        x_data = tf.cast(x_data,tf.float64)
        
        delta_W = tf.random.normal(shape = (n_samples,n_dim),mean = 0.0, stddev = np.sqrt(h))
        delta_W = tf.reshape(delta_W,shape = (n_samples,n_dim,-1))
        delta_W = tf.cast(delta_W,tf.float64)
        
        Drift = np.apply_along_axis(self.drift,1,np.array(x_data))
        Drift = tf.reshape(Drift,shape = (n_samples,n_dim))
        Drift = tf.cast(Drift,tf.float64)
        
        Diffusitivity = np.apply_along_axis(self.diffusitivity,1,np.array(x_data))
        Diffusitivity = tf.constant(Diffusitivity)
        Diffusitivity = tf.cast(Diffusitivity,tf.float64) 
        Diffusitivity = tf.matmul(Diffusitivity,delta_W)
        Diffusitivity = tf.reshape(Diffusitivity,shape = (n_samples,n_dim))
        
        y_data = np.float64(x_data) + np.float64(Drift*h) + Diffusitivity
        return y_data
        


def euler_sde_net(model,u0,delta_t,N,n_dim):
    """Multi-step prediction of an SDE system using neural network
    """
    t = np.arange(N)
    t = t*delta_t
    u = np.zeros((N,n_dim))
    u[0,:] = u0
    for k in range(N-1):
        uk = tf.reshape(u[k,:],shape = (1,-1))
        u[k + 1,:] = model.predict(uk,dt =delta_t) 
    return t,u



def euler_sde(f,sigma,u0,delta_t,N,n_dim):
    """Multi-step prediction of an SDE system using Euler-Maruyama scheme
    """
    t = np.arange(N)
    t = t*delta_t
    u = np.zeros((N,n_dim))
    u[0,:] = u0
    for k in range(N-1):
        K1 = delta_t *f(u[k,:])
        delta_W = np.random.normal(loc = 0.0, scale = np.sqrt(delta_t), size = (n_dim,1))
        K2 = np.dot(sigma(u[k,:]),delta_W)
        u[k+1,:] = u[k,:] +K1 + K2.ravel()
    return t,u    



def euler_net(model,u0,delta_t,N,n_dim):
    """Multi-step prediction of an ODE system using neural network
    """
    t = np.arange(N)
    t = t*delta_t
    u = np.zeros((N,n_dim))
    u[0,:] = u0
    for k in range(N-1):
        uk = tf.reshape(u[k,:],shape = (1,-1))
        u[k + 1,:] = model.predict(uk,verbose = 0) 
    return t,u


def euler(f, u0,t_span,N,n_dim):
    """Multi-step prediction of an ODE system using Euler scheme
    """
    t = np.linspace(t_span[0], t_span[1],N)
    u = np.zeros((N,n_dim))
    u[0,:] = u0
    delta_t = t[1] - t[0]
    for (n, t_n) in enumerate(t[:-1]):
        K1 = delta_t * f(u[n,:])
        u[n + 1,:] = u[n,:] + K1        
    return t, u