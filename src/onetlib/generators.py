import abc
import numpy as np
import pickle
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class TimeSeriesDataGenerator(abc.ABC):
    """Abstract class for input-output time series generators

    It is required to implement the ``generate_data`` method
    which should generate tuple (x, y), each of first dimension
    equal to n_samples
    """
    @abc.abstractmethod
    def generate_data(self, n_samples, **kwargs):
        return


class ForcedODEGenerator(TimeSeriesDataGenerator):
    """Abstract class for forced ODE generators

    Uses the scipy ``solve_ivp`` method to produce input-output datasets
    Subclasses should implement the follow methods:
    * ``generate_initial_conditions``
    * ``f``
    * ``compute_outputs``
    * ``generate_input_functions``

    See individual method docstrings for more information
    """
    def __init__(self, T, n_t):
        """Initializer

        :param T: Terminal Time
        :type T: float
        :param n_t: number of time steps
        :type n_t: int
        """
        self.T = T
        self.n_t = n_t

    @abc.abstractmethod
    def generate_initial_conditions(self, n_samples):
        """Generate initial conditions for ODE

        :param n_samples: number of samples to generate
        :type n_samples: int
        :return: tensors of initial conditions of shape
                 [n_samples, n_dim]
        :rtype: ndarray
        """
        return

    @abc.abstractmethod
    def f(self, states, inputs):
        """Computes the RHS of ODE

        Computes f(h, x), where dh/dt = f(h, x) is the equation

        :param states: states h [n_samples, n_h_dim]
        :type states: ndarray
        :param inputs: inputs x [n_samples, n_x_dim]
        :type inputs: ndarray
        :return: next state [n_samples, n_h_dim]
        :rtype: ndarray
        """
        return states

    @abc.abstractmethod
    def compute_outputs(self, states):
        """Computes the outputs

        Computes an observation function
        y_hat = g(h)

        :param states: states h [n_samples, n_h_dim]
        :type states: ndarray
        :return: next state [n_samples, n_h_dim]
        :rtype: ndarray
        """
        return

    @abc.abstractmethod
    def generate_input_functions(self, n_samples, T, n_t):
        """Generate input function

        :param n_samples: number of samples
        :type n_samples: int
        :param T: terminal time
        :type T: float
        :param n_t: number of time steps
        :type n_t: int
        :return: function which returns t->[n_samples]
        :rtype: function
        """
        return

    def generate_data(self, n_samples, T=None, n_t=None, input_func=None):
        """Generate data

        Generates an input-output time series dataset (x, y) with
        * x[n_samples, n_t, n_x_dim]
        * y[n_samples, n_t, n_y_dim]

        The n_x_dim is given by the method ``generate_initial_conditions``
        The n_y_dim is given by the method ``compute_outputs``

        If ``T`` and ``n_t`` are ``None``, then uses the default values
        from class instance attributes. Otherwise, use the input values

        FIXME: Update docstring wrt input_func, if necessary

        :param n_samples: number of samples
        :type n_samples: int
        :param T: terminal time, defaults to None
        :type T: float, optional
        :param n_t: number of time steps, defaults to None
        :type n_t: int, optional
        :return: (inputs, outputs), each of shape [n_samples, n_t, n_x_dim]
        :rtype: tuple
        """
        # Prepare inputs
        if T is None:
            T = self.T
        if n_t is None:
            n_t = self.n_t
        if input_func is None:
            input_func = self.generate_input_functions(n_samples=n_samples,
                                                       T=T,
                                                       n_t=n_t)
        t_grid = np.linspace(0, T, n_t)
        # TODO: Vectorize if performance critical
        inputs = np.asarray([input_func(t) for t in t_grid]).transpose(1, 0, 2)

        # Solve IVP to compute state
        def fun(t, state):
            state = state.reshape(n_samples, -1)
            f_state = self.f(state, input_func(t))
            return f_state.reshape(-1)

        init_conds = self.generate_initial_conditions(n_samples=n_samples)
        results = solve_ivp(
            fun=fun,
            t_span=[t_grid[0], t_grid[-1]],
            y0=init_conds.reshape(-1),
            t_eval=t_grid,
        )
        state_solution = results.y.reshape(n_samples, -1,
                                           n_t).transpose(0, 2, 1)

        # Compute outputs based on h
        outputs = np.array([
            self.compute_outputs(state_solution[:, k, :]) for k in range(n_t)
        ])
        outputs = outputs.transpose(1, 0, 2)

        return inputs, outputs


class forced(ForcedODEGenerator):
    """Forced Lorenz equation

        x'[t] = sigma * (y[t]-x[t]) + force_scale * u[t]

        y'[t] = x[t] * (r-z[t]) - y[t]

        z'[t] = x[t] * y[t] - b * z[t]

        x[0] = y[0] = z[0] = 0

          * Here, u[t] is a scalar input time series
          * The observations are x[t] / out_scale
    """
    def __init__(self,
                 T,
                 n_t,
                 r=28.0,
                 sigma=10.0,
                 b=8.0 / 3.0,
                 force_scale=100.0,
                 out_scale=10.0):
        """Initializer

        :param T: terminal time
        :type T: float
        :param n_t: number of time steps
        :type n_t: int
        :param r: lorenz r parameter, defaults to 28.0
        :type r: float, optional
        :param sigma: lorenz sigma parameter, defaults to 10.0
        :type sigma: float, optional
        :param b: lorenz b parameter, defaults to 8.0/3.0
        :type b: float, optional
        :param force_scale: force scale, defaults to 100.0
        :type force_scale: float, optional
        :param out_scale: output scale, defaults to 10.0
        :type out_scale: float, optional
        """
        super(ForcedLorenzGenerator, self).__init__(T=T, n_t=n_t)
        self.r = r
        self.sigma = sigma
        self.b = b
        self.force_scale = force_scale
        self.out_scale = out_scale

    def compute_outputs(self, states):
        """Observation of the first state

        outputs is just x[t] / out_scale

        :param states: states
        :type states: ndarray
        :return: outputs
        :rtype: ndarray
        """
        return states[:, 0][:, None] / self.out_scale

    def generate_input_functions(self, n_samples, T, n_t):
        """Generate input function

        returns the function cos(w*t)
        where w is uniform random variable in [0, 2pi]

        :param n_samples: number of samples
        :type n_samples: int
        :return: input generating function
        :rtype: function
        """
        omega = np.random.uniform(low=0, high=2 * np.pi, size=(n_samples, 1))
        return lambda t: np.cos(omega * t)

    def generate_initial_conditions(self, n_samples):
        """Generate initial condtions

        Generates zero initial condition

        :param n_samples: number of samples
        :type n_samples: int
        :return: array of zeros
        :rtype: ndarray
        """
        return np.zeros((n_samples, 3))

    def f(self, states, inputs):
        """Lorenz right hand side

        :param states: states h [n_samples, n_h_dim]
        :type states: ndarray
        :param inputs: inputs x [n_samples, n_x_dim]
        :type inputs: ndarray
        :return: next state [n_samples, n_h_dim]
        :rtype: ndarray
        """
        x, y, z = states[:, 0], states[:, 1], states[:, 2]
        s = self.sigma
        r = self.r
        b = self.b
        inputs = np.squeeze(inputs)
        return np.column_stack([
            s * (y - x) + self.force_scale * inputs, x * (r - z) - y,
            x * y - b * z
        ])


class ForcedLorenzGeneratorV2(ForcedLorenzGenerator):
    """Forced Lorenz equation with forcing entering r

        x'[t] = sigma * (y[t] - x[t])

        y'[t] = x[t] * (r*u[t] - z[t]) - y[t]

        z'[t] = x[t] * y[t] - b * z[t]

        x[0] = y[0] = z[0] = 1

          * Here, u[t] is a scalar input time series
          * The observations are x[t] / out_scale
    """
    def f(self, states, inputs):
        x, y, z = states[:, 0], states[:, 1], states[:, 2]
        s = self.sigma
        r = self.r
        b = self.b
        inputs = np.squeeze(inputs)
        return np.column_stack([
            s * (y - x), x * (r * self.force_scale * inputs - z) - y,
            x * y - b * z
        ])

    def generate_input_functions(self, n_samples, T, n_t):
        """Generate input function

        returns the function (cos(w*t) + 1)/2
        where w is uniform random variable in [0, 2pi]

        :param n_samples: number of samples
        :type n_samples: int
        :return: input generating function
        :rtype: function
        """
        omega = np.random.uniform(low=0, high=2 * np.pi, size=(n_samples, 1))
        return lambda t: 0.5 * (1 + np.cos(omega * t))

    def generate_initial_conditions(self, n_samples):
        """Generate initial condtions

        Generates identity initial condition

        :param n_samples: number of samples
        :type n_samples: int
        :return: array of zeros
        :rtype: ndarray
        """
        return np.ones((n_samples, 3))


class ForcedLinearODEGenerator(ForcedODEGenerator):
    """Forced Linear ODE Generator

    Generates the dynamics
    h'[t] = A^T h[t] - M h[t] + x[t] * [0, ... , 0, 1]
    h[0] = 0

    A is an anti-symmetric matrix
    M is a symmetric positive semi-definite matrix
    """
    def __init__(self, T, n_t, n_dim, out_scale=50.0):
        """Initializer

        :param T: terminal time
        :type T: float
        :param n_t: time steps
        :type n_t: int
        :param n_dim: hidden dim
        :type n_dim: int
        """
        super(ForcedLinearODEGenerator, self).__init__(T=T, n_t=n_t)
        self.n_dim = n_dim
        self.out_scale = out_scale
        B = np.random.normal(size=(n_dim, n_dim))
        self.A = B.T - B
        N = np.random.normal(size=(n_dim, n_dim))
        self.M = (N.T @ N) / n_dim

    def compute_outputs(self, states):
        """Observation of the first state

        output = out_scale * h[0]

        :param states: states
        :type states: ndarray
        :return: outputs
        :rtype: ndarray
        """
        return states[:, 0][:, None] * self.out_scale

    def generate_input_functions(self, n_samples, T, n_t):
        """Generate input function

        returns the function
          x(t) = s * w(t) * e^{-t}
        where
        - w(t) is a piece-wise constant interpolation of a white noise
        - s is a uniform random scale with distribution U[0, 10]
        - exponential decay is to check dissipative behavior without noise

        :param n_samples: number of samples
        :type n_samples: int
        :return: input generating function
        :rtype: function
        """
        t_grid = np.linspace(0, T, n_t)
        x_grid = np.random.normal(size=(n_samples, n_t))
        scale_grid = np.random.uniform(0.0, 10.0, size=(n_samples, 1))
        func = interp1d(t_grid, x_grid, kind=0, fill_value='extrapolate')
        return lambda t: func(t).reshape(n_samples, 1) * np.exp(-1.0 * t
                                                                ) * scale_grid

    def generate_initial_conditions(self, n_samples):
        """Generate initial condtions

        Generates zero initial condition

        :param n_samples: number of samples
        :type n_samples: int
        :return: array of zeros
        :rtype: ndarray
        """
        return np.zeros((n_samples, self.n_dim))

    def f(self, states, inputs):
        """Linear ODE right hand side

        :param states: states h [n_samples, n_h_dim]
        :type states: ndarray
        :param inputs: inputs x [n_samples, n_x_dim]
        :type inputs: ndarray
        :return: next state [n_samples, n_h_dim]
        :rtype: ndarray
        """
        inputs = np.squeeze(inputs)

        rhs = states @ self.A - states @ self.M
        rhs[:, -1] = rhs[:, -1] + inputs
        return rhs


# ------------------------------------------------------------------ #
#                           Lorenz96 Model                           #
# ------------------------------------------------------------------ #


class ForcedLorenz96Generator(ForcedODEGenerator):
    def __init__(self, T, n_t, n_dim):
        super(ForcedLorenz96Generator, self).__init__(T=T, n_t=n_t)
        self.h_init = np.random.normal(size=(n_dim, ))
        self.n_dim = n_dim
        self.output_vec = np.random.normal(size=(n_dim, ))

    def compute_outputs(self, states):
        """Compute outputs

        average kinetic energy or linear observation
        """
        # return states @ self.output_vec.reshape(-1, 1) * 1.0
        return np.mean(states**2, axis=1, keepdims=True) * 0.1
        # FIXME: divide by 10 to make order 1

    def generate_initial_conditions(self, n_samples):
        # ones = np.ones((n_samples, self.n_dim))
        # return self.h_init.reshape(1, -1) * ones
        zeros = np.zeros((n_samples, self.n_dim))
        return self.h_init.reshape(1, -1) * zeros

    def generate_input_functions(self, n_samples, T, n_t):
        # Base Function
        F_0 = 8.0  # Base force
        F_range = (-2.0, 2.0)  # range of force fluctuation
        force_dt = 0.2  # frequency of applying a forceo
        n_t_interp = round(T / force_dt)
        t_grid = np.linspace(0, T, n_t_interp)
        x_grid = np.random.uniform(*F_range, size=(n_samples, n_t_interp))
        interp_func = interp1d(t_grid,
                               x_grid,
                               kind=0,
                               fill_value='extrapolate')

        # Apply Random Transformations
        scale_grid = np.random.uniform(low=0, high=2, size=(n_samples, 1))
        decay_grid = np.random.uniform(low=0, high=0.5, size=(n_samples, 1))
        osci_grid = np.random.uniform(low=0, high=10, size=(n_samples, 1))

        def func(t):
            interp_value = F_0 + interp_func(t)
            interp_value = interp_value.reshape(n_samples, 1)
            return interp_value * np.exp(
                -decay_grid * t) * scale_grid * np.cos(osci_grid * t)

        return func

    def f(self, states, inputs):
        h_plus_one = np.roll(states, -1, axis=1)
        h_minus_one = np.roll(states, 1, axis=1)
        h_minus_two = np.roll(states, 2, axis=1)
        rhs = (h_plus_one - h_minus_two) * h_minus_one - states + inputs
        return rhs * 5.0  # FIXME: patch-fix for timescale


# ------------------------------------------------------------------ #
#                          Loader Generator                          #
# ------------------------------------------------------------------ #


class LoaderGenerator(TimeSeriesDataGenerator):
    """Loads Training Data from File
    """

    def __init__(self, load_path, test_size=0.2):
        with open(load_path, 'rb') as file:
            self.data = pickle.load(file)
        self.n_data = self.data[0].shape[0]
        self.n_test = int(test_size * self.n_data)
        self.n_train = self.n_data - self.n_test

    def generate_data(self, n_samples=None, train=True):
        assert n_samples is None
        x, y = self.data
        if train:
            return x[:self.n_train], y[:self.n_train]
        else:
            return x[self.n_train:], y[self.n_train:]
