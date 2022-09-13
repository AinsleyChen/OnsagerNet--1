import onetlib.layers as olayers
from onetlib.layers import BaseLayer
from onetlib.initializers import GlorotUniform
from onetlib.initializers import ScaledGlorotUniform
from onetlib.initializers import FlattenedIdentity
from tensorflow.keras.layers import Dense
import tensorflow as tf

# ------------------------------------------------------------------ #
#                         Potential Networks                         #
# ------------------------------------------------------------------ #


class FCPotentialNet(BaseLayer):
    """Fully Connected Potential Network

    Implements fully connected network based potential function

    V(x) = beta * ||x||^2 + sum_{j=1}^{n_pot} phi_j(x)^2

    Each phi_j is a fully connected network with nodes given by
    ``layer_sizes``
    """
    def __init__(self, layer_sizes=[32], n_pot=32, beta=0.01, **kwargs):
        """Initializer

        :param layer_sizes: size of layers, defaults to [32]
        :type layer_sizes: list, optional
        :param n_pot: number of basis functions, defaults to 32
        :type n_pot: int, optional
        :param beta: regularizer, defaults to 0.01
        :type beta: float, optional
        """
        super(FCPotentialNet, self).__init__(**kwargs)
        self.hidden_layers = [
            Dense(s, kernel_initializer=ScaledGlorotUniform())
            for s in layer_sizes
        ]
        
        self.activation_layers = [olayers.ShiftedRePU() for s in layer_sizes]
        #self.activation_layers = [tf.keras.layers.Activation('tanh') for s in layer_sizes]
        self.output_layer = Dense(n_pot,
                                  kernel_initializer=ScaledGlorotUniform())
        self.regularized_square = olayers.RegularizedSquare(beta=beta)
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.n_pot = n_pot

    def call(self, inputs):
        x = inputs
        for layer, activation in zip(self.hidden_layers,
                                     self.activation_layers):
            x = layer(x)
            x = activation(x)
        outputs = self.output_layer(x)
        outputs = self.regularized_square([inputs, outputs])
        return outputs

    def get_config(self):
        config = super(FCPotentialNet, self).get_config()
        config.update({
            'layer_sizes': self.layer_sizes,
            'beta': self.beta,
            'n_pot': self.n_pot
        })
        return config


# ------------------------------------------------------------------ #
#                Dissipative and Conservative Networks               #
# ------------------------------------------------------------------ #


class FCDissConsNet(BaseLayer):
    """Fully connected dissipation-conservation matrix network

    Implements a fully connected network based matrix-valued function
    on the hidden states h.The outputs are:
    * a symmetric positive semi-definite matrix M(h)
    * an antisymmetric matrix W(h)
    """
    def __init__(self, n_dim, layer_sizes=[32], **kwargs):
        """Initializer

        :param n_dim: hidden dimension
        :type n_dim: int
        :param layer_sizes: size of layers, defaults to [32]
        :type layer_sizes: list, optional
        """
        super(FCDissConsNet, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.n_dim = n_dim
        self.hidden_layers = [
            Dense(s, kernel_initializer=ScaledGlorotUniform())
            for s in layer_sizes
        ]
        self.activation_layers = [olayers.ShiftedRePU() for s in layer_sizes]
       
        self.output_layer = Dense(n_dim * n_dim,
                                  kernel_initializer=ScaledGlorotUniform(),
                                  bias_initializer=FlattenedIdentity())
        self.decomposition_layer = olayers.SymmAntiDecomposition(n_dim=n_dim)

    def call(self, inputs):
        x = inputs
        for layer, activation in zip(self.hidden_layers,
                                     self.activation_layers):
            x = layer(x)
            x = activation(x)
        output = self.output_layer(x)
        m, w = self.decomposition_layer(output)
        return m, w

    def get_config(self):
        config = super(FCDissConsNet, self).get_config()
        config.update({'layer_sizes': self.layer_sizes, 'n_dim': self.n_dim})
        return config


# ------------------------------------------------------------------ #
#                      External Forcing Networks                     #
# ------------------------------------------------------------------ #




class LinearForcingNet(BaseLayer):
    """Linear forcing network

    Implements a linear (affine) forcing network

    F(x) = W x + b
    """
    def __init__(self, n_dim, **kwargs):
        """Initializer

        :param n_dim: hidden dimension
        :type n_dim: int
        """
        super(LinearForcingNet, self).__init__(**kwargs)
        self.linear_layer = Dense(n_dim,
                                  kernel_initializer=GlorotUniform())
        self.n_dim = n_dim

    def call(self, inputs):
        return self.linear_layer(inputs)

    def get_config(self):
        config = super(LinearForcingNet, self).get_config()
        config.update({'n_dim': self.n_dim})
        return config

      
class FCForcingNet(LinearForcingNet):

    def __init__(self, n_dim, layer_sizes=[32], **kwargs):

        super(FCForcingNet, self).__init__(n_dim, **kwargs)
        self.layer_sizes = layer_sizes
        self.n_dim = n_dim
        # self.hidden_layers = [
        #     Dense(s, activation='relu', kernel_initializer=ScaledGlorotUniform())
        #     for s in layer_sizes
        # ]
        self.hidden_layers = [
            Dense(s)
            for s in layer_sizes
        ]

        self.activation_layers = [olayers.ShiftedRePU() for s in layer_sizes]
        # self.output_layer = Dense(n_dim,
        #                           kernel_initializer=GlorotUniform())
        self.output_layer = Dense(n_dim)

    def call(self, inputs):
        x = inputs
        for layer, activation in zip(self.hidden_layers,
                                     self.activation_layers):
            x = layer(x)
            x = activation(x)

        output = self.output_layer(x)
        return output
    

# class FCForcingNet(LinearForcingNet):

#     def __init__(self, n_dim, layer_sizes=[32], **kwargs):
#         """Initializer

#         :param n_dim: hidden dimension
#         :type n_dim: int
#         :param layer_sizes: size of layers, defaults to [32]
#         :type layer_sizes: list, optional
#         """
#         super(FCForcingNet, self).__init__(**kwargs)
#         self.layer_sizes = layer_sizes
#         self.n_dim = n_dim
#         self.hidden_layers = [
#             Dense(s, kernel_initializer=ScaledGlorotUniform())
#             for s in layer_sizes
#         ]
#         self.activation_layers = [olayers.ShiftedRePU() for s in layer_sizes]
#         self.output_layer = Dense(n_dim,
#                                   kernel_initializer=GlorotUniform())

#     def call(self, inputs):
#         x = inputs
#         for layer, activation in zip(self.hidden_layers,
#                                      self.activation_layers):
#             x = layer(x)
#             x = activation(x)
#         output = self.output_layer(x)
#         return output

#     def get_config(self):
#         config = super(FCForcingNet, self).get_config()
#         config.update({'layer_sizes': self.layer_sizes, 'n_dim': self.n_dim})
#         return config


# ------------------------------------------------------------------ #
#                         OnsagerNet Networks                        #
# ------------------------------------------------------------------ #


class OnsagerNet(BaseLayer):
    """OnsagerNet Architecture

    Implements the OnsagerNet architecture

    h -> - (M(h) + W(h) + alpha*I) nabla V(h) + F(x)

    where
    * h is the hidden state, x is the input
    * M(h), W(h) are from the dissconv_net
    * V(h) is from the potential_net
    * F(x) is from the forcing_net
    """
    external_layers = {'potential_net', 'dissconv_net', 'forcing_net'}

    def __init__(self,
                 n_dim,
                 potential_net,
                 dissconv_net,
                 forcing_net,
                 alpha=0.01,
                 **kwargs):
        """Initializer

        :param n_dim: hidden dimension
        :type n_dim: int
        :param potential_net: potential network
        :type potential_net: BaseLayer
        :param dissconv_net: dissipative-conservative network
        :type dissconv_net: BaseLayer
        :param forcing_net: forcing from hidden states
        :type forcing_net: BaseLayer
        :param alpha: regularizer, defaults to 0.01
        :type alpha: float, optional
        """
        super(OnsagerNet, self).__init__(**kwargs)
        self.n_dim = n_dim
        self.alpha = alpha
        self.potential_net = potential_net
        self.pot_grad_net = olayers.GradientLayer(func=self.potential_net)
        self.dissconv_net = dissconv_net
        self.forcing_net = forcing_net
        self.combination_layer = olayers.OnsagerCombination(alpha=alpha)

    def call(self, inputs):
        h = inputs
        g = self.pot_grad_net(h)
        f = self.forcing_net(h)
        m, w = self.dissconv_net(h)
        return self.combination_layer([m, w, g, f])

    # def call(self, inputs):
    #     x, h = inputs
    #     g = self.pot_grad_net(h)
    #     f = self.forcing_net(x)
    #     m, w = self.dissconv_net(h)
    #     return self.combination_layer([m, w, g, f])

    def get_config(self):
        config = super(OnsagerNet, self).get_config()
        config.update({'n_dim': self.n_dim, 'alpha': self.alpha})
        return config

# ------------------------------------------------------------------ #
#                   OnsagerNet Euler Integrator                      #
# ------------------------------------------------------------------ #

class EulerOnsagerLayer(BaseLayer):
    """Euler integrator based OnsagerRNN Cell

    Implements the OnsagerRNN Cell which computes

    (x, h) -> h + dt * OnsagerNet(x, h)

    This is to be used in conjection with ``tf.keras.layers.RNN``
    as a custom cell.
    """
    external_layers = ['onsager_layer']

    def __init__(self, n_dim, onsager_layer, dt, **kwargs):
        """Initializer

        :param n_dim: hidden dimension
        :type n_dim: int
        :param onsager_layer: OnsagerNet layer
        :type onsager_layer: BaseLayer
        :param dt: time step
        :type dt: float
        """
        super(EulerOnsagerLayer, self).__init__(**kwargs)
        self.n_dim = n_dim
        self.dt = dt
        self.onsager_layer = onsager_layer
        self.integration_layer = olayers.FwEulerLayer(dt=dt)
        self.state_size = n_dim  
        self.output_size = n_dim 

    def call(self, inputs):
        # states = states[0]
        rhs = self.onsager_layer(inputs)
        next_states = self.integration_layer([inputs, rhs])
        return next_states

    def get_config(self):
        config = super(EulerOnsagerLayer, self).get_config()
        config.update({'n_dim': self.n_dim, 'dt': self.dt})
        return config

    
    
# ------------------------------------------------------------------ #
#                   Stochastic OnsagerNet                            #
# ------------------------------------------------------------------ #
import tensorflow as tf
import numpy as np

class ConstantLayer(tf.keras.layers.Layer):
    """Generate constants when the diffusion term 
    of a SDE is chosen to be mode 'constant_diagonal'
    """
    def __init__(self, n_dim):
        """Initializer
        
        :param n_dim: hidden dimension
        :type n_dim: int
        """
        super(ConstantLayer, self).__init__()
        self.n_dim = n_dim
        
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",shape=[self.n_dim],trainable=True)

    def call(self, inputs):
        # inputs shape: [batchsize,n_dim]
        # output shape: [batchsize,n_dim]
        output = inputs*0 + self.kernel
        return output
    
    def get_config(self):
        config = super(ConstantLayer, self).get_config()
        config.update({'n_dim': self.n_dim})
        return config
    
    
class DiffusitivityNet(BaseLayer):
    """Generate diffusitivity term for SDE
    """
    def __init__(self, n_dim, layer_sizes=[32],mode = 'arbitrary', **kwargs):
        """Initializer
        
        :param n_dim: hidden dimension
        :type n_dim: int
        :param layer_sizes: size of layers, defaults to [32]
        :type layer_sizes: list, optional
        :param mode: the structure of the diffusion term \sigma of a SDE
        :type mode:one string from ['arbitrary', 'constant_diagonal','diagonal'] 
        :'arbitrary': no restriction on diffusiong term
        :'diagonal': diffusion term is a diagonal matrix
        :'constant_diagonal':diffusion term in a constant and diagonal matrix
        """
        super(DiffusitivityNet, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.n_dim = n_dim
        self.hidden_layers = [
            Dense(s)
            for s in layer_sizes
        ]

        self.activation_layers = [olayers.ShiftedRePU() for s in layer_sizes]

        if mode not in ['arbitrary', 'constant_diagonal','diagonal']:
            logging.warning('mode %s is unknown, '
                      'fallback to auto mode.', self.mode)
            mode = 'arbitrary'
        self.mode = mode
        
        if mode == 'arbitrary':
            self.output_layer = Dense(n_dim*n_dim)
        elif mode =='diagonal':
            self.output_layer = Dense(n_dim)
        elif mode == 'constant_diagonal':
            self.output_layer = ConstantLayer(self.n_dim)

            
    def call(self, inputs):
        x = inputs
        
        if self.mode=='arbitrary':
            output_layer = Dense(self.n_dim*self.n_dim)
            for layer, activation in zip(self.hidden_layers,
                                         self.activation_layers):
                x = layer(x)
                x = activation(x)

            output = self.output_layer(x)
             
        elif self.mode == 'diagonal':
            for layer, activation in zip(self.hidden_layers,
                                         self.activation_layers):
                x = layer(x)
                x = activation(x)
                
            output = self.output_layer(x)
            output = tf.linalg.diag(output)
            output = tf.reshape(output, shape = (-1,self.n_dim*self.n_dim))
    
        elif self.mode == 'constant_diagonal':
            # shape:[batchsize,n_dim]
            output = self.output_layer(x)  
            # shape:[batchsize,n_dim,n_dim]  create constant diagonal matrix
            output = tf.linalg.diag(output) 
            # reshape : [batchsize, n_dim*n_dim] 
            output = tf.reshape(output, shape = (-1,self.n_dim*self.n_dim))  
           
        return output

    def get_config(self):
        config = super(DiffusitivityNet, self).get_config()
        config.update({'layer_sizes': self.layer_sizes, 'n_dim': self.n_dim,'mode':self.mode})
        return config

    
    

# ------------------------------------------------------------------ #
#                     Additional Forcing Networks                    #
# ------------------------------------------------------------------ #  
class ZeroForcingNet(BaseLayer):
    """Zero forcing network
    """
    def __init__(self, n_dim, **kwargs):
        """Initializer

        :param n_dim: hidden dimension
        :type n_dim: int
        """
        super(ZeroForcingNet, self).__init__(**kwargs)
        self.n_dim = n_dim

    def call(self, inputs):
        kernel = tf.zeros((self.n_dim,self.n_dim))
        return tf.matmul(inputs,kernel)

    def get_config(self):
        config = super(ZeroForcingNet, self).get_config()
        config.update({'n_dim': self.n_dim})
        return config


# ------------------------------------------------------------------ #
#                      Custom Loss Function for SDE                  #
# ------------------------------------------------------------------ #    

class CustomEulerLoss(tf.keras.losses.Loss):
    """Custom Loss for Stochastic OnsagerNet
    """
    def __init__(self,epsilon,n_features,delta_t):
        """Initializer

        :param epsilon: the normalizing parameter
        :type epsilon: float(often small)
        :param n_features: hidden dimension
        :type n_features: int
        """
        super().__init__() 
        self.epsilon = epsilon
        self.n_features = n_features
        delta_t = tf.cast(delta_t,tf.float64)
        self.delta_t = delta_t
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true,tf.float64)
    
        x = y_pred[:,0:self.n_features]
        x = tf.cast(x,tf.float64)
        
        # the drift term of SDE
        drift = y_pred[:,self.n_features:2*self.n_features]
        drift = tf.cast(drift,tf.float64)
        
        # the diffusion term of a SDE
        sigma = y_pred[:,2*self.n_features:]
        sigma = tf.cast(sigma,tf.float64)
        sigma_delta_t = sigma *tf.math.sqrt(self.delta_t)
        sigma_delta_t = tf.reshape(sigma_delta_t,shape = (-1,self.n_features,self.n_features))
        sigma_delta_t = tf.cast(sigma_delta_t,tf.float64)
        
        # ensure the existense of the inverse of Sigma
        B = tf.eye(self.n_features)*self.epsilon
        B = tf.cast(B,tf.float64)
        sigma_delta_t = sigma_delta_t + B

        Sigma = tf.einsum('ijk,ilk->ijl',sigma_delta_t,sigma_delta_t)
        Sigma = tf.cast(Sigma,tf.float64)
        Sigma = tf.reshape(Sigma,shape = (-1,self.n_features,self.n_features))
        Sigma_inv = tf.linalg.inv(Sigma)

        X = y_true - x - drift*self.delta_t
        a1 = tf.einsum('ij,ijk,ik->i',X,Sigma_inv,X)
        a1 = tf.reshape(a1,shape = (-1,1))

        a2 = tf.linalg.logdet(Sigma)
        a2 = tf.cast(a2,tf.float64)
        a2 = tf.reshape(a2,(-1,1))

        a3 = self.n_features *tf.math.log(2*np.pi)
        a3 = tf.cast(a3,tf.float64)

        return tf.reduce_mean(a1+a2+a3)
    
# ------------------------------------------------------------------ #
#        Custom OnsargerNet for deterministic and stochastic system  #
# ------------------------------------------------------------------ #    
    
class ODEOnsagerNet(tf.keras.Model):
    """Custom OnsargerNet for deterministic OnsagerNet
    """
    def __init__(self,n_features,delta_t,OnsagerNet,**kwargs):
        super(ODEOnsagerNet,self).__init__(**kwargs)
        self.n_features = n_features
        self.delta_t = delta_t
        self.onsager_rhs = OnsagerNet
    
    def call(self, inputs):
        onsager_pred = EulerOnsagerLayer(n_dim = self.n_features,
                                         onsager_layer = self.onsager_rhs,
                                         dt = self.delta_t)
        return onsager_pred(inputs)
    
    def get_config(self):
        config = super(ODEOnsagerNet, self).get_config()
        config.update({'n_features': self.n_features, 
                       'delta_t': self.delta_t,
                       'OnsargerNet':self.onsager_rhs})
        return config
    

class SDEOnsagerNet(ODEOnsagerNet):
    """Custom OnsargerNet for stochastic OnsagerNet
    """
    def __init__(self,n_features,delta_t,OnsagerNet,**kwargs):
        super(SDEOnsagerNet,self).__init__(n_features,delta_t,OnsagerNet,**kwargs)
    
    def call(self, inputs):
        return self.onsager_rhs(inputs)
    
    def predict(self,x,dt):
        x = tf.cast(x,tf.float64)
        y_pred = self.call(x)
        drift = y_pred[:,self.n_features:2*self.n_features]
        drift = tf.cast(drift,tf.float64)
        
        sigma = y_pred[:,2*self.n_features:]
        sigma = tf.reshape(sigma,shape = (-1,self.n_features,self.n_features))
        sigma = tf.cast(sigma, tf.float64)
        delta_W = tf.random.normal(shape = (x.shape[0],self.n_features),
                                   mean = 0.0, stddev = np.sqrt(dt))
        delta_W = tf.cast(delta_W,tf.float64)
    
        return x + drift*dt + tf.einsum('ijk,ik->ij',sigma,delta_W)
    
    def get_config(self):
        config = super(SDEOnsagerNet, self).get_config()
        config.update({'n_features': self.n_features, 
                       'delta_t': self.delta_t,
                       'OnsargerNet':self.onsager_rhs})
        return config