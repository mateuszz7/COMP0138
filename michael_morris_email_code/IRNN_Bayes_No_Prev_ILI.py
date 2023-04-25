import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tqdm

from scipy.stats import norm

from GRU_cell_variational import *
from Dense_Variational_Reparam import *

tfd = tfp.distributions

def get_cal(true, mean, std):
    def calc_in_range(mean, std, true, z):
        return(np.sum(np.logical_and((mean - z * std) < true,
                                        (mean + z * std) > true)) / mean.shape[0])

    stds = np.linspace(0, 3, 50)
    freq = np.zeros(stds.shape)
    probs = norm.cdf(stds) - norm.cdf(-stds)

    for idx, z in enumerate(stds):
        freq[idx] = calc_in_range(mean, std, true, z)
    
    return probs, freq

def random_gaussian_initializer(shape, dtype):
    if shape[0] == 2:
        n = shape[1:]
        loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
        loc = tf.Variable(
            initial_value=loc_norm(shape=n, dtype=dtype)
        )
        scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
        scale = tf.Variable(
            initial_value=scale_norm(shape=n, dtype=dtype)
        )
        init = tf.concat([tf.expand_dims(loc, 0), tf.expand_dims(scale,0)], 0)
    # if len(shape) >= 3:
    #     n = shape[1:]
    #     loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
    #     loc = tf.Variable(
    #         initial_value=loc_norm(shape=n, dtype=dtype)
    #     )
    #     scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
    #     scale = tf.Variable(
    #         initial_value=scale_norm(shape=n, dtype=dtype)
    #     )
    #     init = tf.concat([tf.expand_dims(loc, 0), tf.expand_dims(scale,0)], 0)
    else:
        n = (shape[0], int(shape[-1]/2))
        loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
        init = tf.Variable(
            initial_value=loc_norm(shape=shape, dtype=dtype)
        )
        # scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
        # scale = tf.Variable(
        #     initial_value=scale_norm(shape=n, dtype=dtype)
        # )
        # init = tf.concat([loc, scale], 1)
    return init

class IRNN_Bayes(tf.keras.Model):
    # parameters for data builder
    lag = 14                # delay between Qs and ILI rates
    model_type = 'feedback'
    forecast_type = 'multi' # does the model forecast once (single) or make a series or forecasts? (feedback)
    loss = 'NLL'             # 
    query_forecast = 'True' # does the network forecast queries or just ILI?


    # upper and lower limits for optimization, good hyper parameters defined as standard.
    pbounds = {'rnn_units':(25,125),        # units in rnn layer
               'n_op':(20,100),             # number of outputs (m+1)
               'kl_power':(-3,0),           # KL annealing term = 10^kl_power
               'p_scale_pwr' :(-3,0),       # prior std = 10^p_scale_pwr
               'q_scale_pwr' :(-3,0),       # post std = 10^p_scale_pwr
               'op_scale_pwr':(-3,0),       # scaling factor for output
               'epochs':(30,200),           # epochs to train for
               'lr_power':(-4, -2),         # learning rate = 10^lr_power
               }

    def __init__(self, 
                 rnn_units=16, 
                 n_op = 1, 
                 kl_power=-2.0,
                 p_scale_pwr = -2.0,
                 q_scale_pwr = -2.0,
                 op_scale_pwr= -2.0,
                 window_size = 47, 
                 gamma=28, 
                 lag = 7, 
                 n_batches=249, 
                 project=False, 
                 n_samples=3,
                 sampling='once',
                 kl_use_exact=True, **kwargs):
        super().__init__()
        rnn_units = int(rnn_units)
        self.n_op = int(n_op)
        self.kl_weight = np.power(10.0, kl_power)
        p_scale = np.power(10.0, p_scale_pwr)
        q_scale = np.power(10.0, q_scale_pwr)
        op_scale = np.power(10.0, op_scale_pwr)
        self.window_size = int(window_size)
        self.gamma = int(gamma)
        self.lag=lag
        self.n_batches = n_batches
        self.project = project
        n_samples = n_samples 
        self.sampling=sampling
        self.kl_use_exact = kl_use_exact
        self.kl_d = 0.
        

        def rnn_posterior(shape, name, initializer, scale = None, project = False, n_samples = n_samples, regularizer=None, constraint=None, cashing_device=None, dtype=tf.float32):
            if scale == None:
                # inspired by glorot uniform, doesn't work very well
                scale = tf.math.sqrt(2/(shape[0] + shape[1]))
            
            c = np.log(np.expm1(1.))
            if not project:
                posterior_model = tf.keras.Sequential([
                    tfp.layers.VariableLayer((2, ) + shape, dtype=dtype, trainable=True,
                                                initializer=lambda shape, dtype: initializer(shape, dtype),
                                                regularizer = regularizer,
                                                name=name
                                                ),

                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.MultivariateNormalDiag(loc=t[0],
                                                    scale_diag=1e-5 + q_scale*tf.nn.softplus(c + t[1])),
                    ))
                ])
            else:
                posterior_model = tf.keras.Sequential([
                    tfp.layers.VariableLayer((2, ) + shape, dtype=dtype, trainable=True,
                                                initializer=lambda shape, dtype: initializer(shape, dtype),
                                                regularizer = regularizer,
                                                name=name
                                                ),

                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.MultivariateNormalDiag(loc=tf.repeat(tf.expand_dims(t[0], 0), n_samples, 0),
                                                    scale_diag=1e-5 + q_scale*tf.nn.softplus(c + tf.repeat(tf.expand_dims(t[1], 0), n_samples, 0),)),
                    ))
                ])   
            return posterior_model

        def rnn_prior(shape, name, initializer,  scale = None, project = False,  n_samples = n_samples, regularizer=None, constraint=None, cashing_device=None, dtype=tf.float32):
            if scale == None:
                # inspired by glorot uniform, doesn't work very well
                scale=tf.math.sqrt(2/(shape[0] + shape[1]))
            if not project:
                prior_model = tf.keras.Sequential([
                    tfp.layers.VariableLayer(shape, dtype=dtype, trainable=True,
                                                initializer=initializer,
                                                regularizer = regularizer,
                                                name=name
                                                ),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.MultivariateNormalDiag(loc = t, scale_diag = p_scale*tf.ones(shape))
                    ))
                ])
            else:
                prior_model = tf.keras.Sequential([
                    tfp.layers.VariableLayer(shape, dtype=dtype, trainable=True,
                                                initializer=lambda shape, dtype: initializer(shape, dtype),
                                                regularizer = regularizer,
                                                name=name
                                                ),

                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.MultivariateNormalDiag(loc=tf.repeat(tf.expand_dims(t, 0), n_samples, 0),
                                                    scale_diag = p_scale*tf.ones((n_samples, ) + shape))
                    ))
                ])   
            return prior_model
        
        def mean0_prior(shape, name, initializer,  scale = None, project = False,  n_samples = n_samples, regularizer=None, constraint=None, cashing_device=None, dtype=tf.float32):
            if scale == None:
                # inspired by glorot uniform, doesn't work very well
                scale=tf.math.sqrt(2/(shape[0] + shape[1]))
            if not project:
                prior_model = tf.keras.Sequential([
                    tfp.layers.VariableLayer(shape, dtype=dtype, trainable=True,
                                                initializer=initializer,
                                                regularizer = regularizer,
                                                name=name
                                                ),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.MultivariateNormalDiag(loc = tf.zeros(shape), scale_diag =p_scale*tf.nn.softplus(c + t))
                    ))
                ])
            else:
                prior_model = tf.keras.Sequential([
                    tfp.layers.VariableLayer(shape, dtype=dtype, trainable=True,
                                                initializer=lambda shape, dtype: initializer(shape, dtype),
                                                regularizer = regularizer,
                                                name=name
                                                ),

                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.MultivariateNormalDiag(loc=tf.zeros((n_samples, ) + shape),
                                                    scale_diag=1e-5 + p_scale*tf.nn.softplus(c + tf.repeat(tf.expand_dims(t, 0), n_samples, 0),)),
                    ))
                ])  
            return prior_model
        
        def mixture_prior(shape, name, std1 = 0.1, std2 = 1e-3, pi=0.3, scale = None, project = False, n_samples = n_samples, regularizer=None, constraint=None, cashing_device=None, dtype=tf.float32, **kwargs):
            def model(a):
                return tfd.Mixture(
                            cat=tfd.Categorical(probs=[pi, 1-pi]),
                            components = [
                                tfd.Normal(loc = 0., scale = std1),
                                tfd.Normal(loc = 0., scale = std2)
                                ]
                            )
            return model

        def hierarchical_prior(shape, name, initializer, pi=0.5, scale = None, project = False, n_samples = n_samples, regularizer=None, constraint=None, cashing_device=None, dtype=tf.float32):
            if scale == None:
                scale = tf.math.sqrt(2/(shape[0] + shape[1]))

            c = np.log(np.expm1(1.))

            prior_model = tf.keras.Sequential([
            tfp.layers.VariableLayer((2, ) + shape, dtype=dtype, trainable=True,
                                        initializer=initializer,
                                        regularizer = regularizer,
                                        name=name
                                        ),
            tfp.layers.DistributionLambda(lambda t: tfd.Mixture(
                cat=tfd.Categorical(
                    probs=tf.tile(tf.stack([[pi], [1-pi]], axis=-1), [shape[0], 1])),
                    components = [
                        tfd.MultivariateNormalDiag(
                            loc = tf.zeros(t[0].shape[1:]), 
                            scale_diag = 1e-5 + q_scale*tf.nn.softplus(c + 25*t[0])),
                        tfd.MultivariateNormalDiag(
                            loc = tf.zeros(t[0].shape[1:]), 
                            scale_diag = 1e-5 + q_scale*tf.nn.softplus(c + t[1]))
                    ]))
            ])
            return prior_model
        
        self.rnn_cell = GRU_Cell_Variational(
            int(rnn_units), 
            kernel_prior_fn = mean0_prior, 
            kernel_posterior_fn = rnn_posterior,
            recurrent_kernel_prior_fn = mean0_prior, 
            recurrent_kernel_posterior_fn = rnn_posterior,
            bias_prior_fn = mean0_prior,
            bias_posterior_fn = rnn_posterior, 
            project = project,
            n_samples = n_samples,
            scale = q_scale,
            kl_weight = self.kl_weight,
            sampling=self.sampling
            
        )

        self.rnn = tf.keras.layers.RNN(self.rnn_cell, return_state=True)

        self.dense_var = DenseVariational_repareterisation(
            units=int(n_op*2),
            last_dim=int(rnn_units),
            make_posterior_fn=rnn_posterior,
            make_prior_fn=mean0_prior,
            initializer=random_gaussian_initializer,
            scale = q_scale,
            project = project,
            n_samples=n_samples,
            kl_weight=self.kl_weight
        )            

        c = np.log(np.expm1(1.))
        self.DistributionLambda = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :self.n_op],
                                scale=1e-5 + op_scale*tf.nn.softplus(c + t[..., self.n_op:])
                                )
            )
        
    def get_kl(self):
        a = self.layers[0]._kernel_posterior(tf.random.normal([5,]))
        b = self.layers[0]._recurrent_kernel_posterior(tf.random.normal([5,]))
        c = self.layers[0]._bias_posterior(tf.random.normal([5,]))
        d = self.layers[2]._posterior(tf.random.normal([5,]))

        e = self.layers[0]._kernel_prior(tf.random.normal([5,]))
        f = self.layers[0]._recurrent_kernel_prior(tf.random.normal([5,]))
        g = self.layers[0]._bias_prior(tf.random.normal([5,]))
        h = self.layers[2]._prior(tf.random.normal([5,]))

        q = tfd.MultivariateNormalDiag(loc = tf.concat([tf.reshape(a.mean(), -1), 
                                                tf.reshape(b.mean(), -1), 
                                                tf.reshape(c.mean(), -1), 
                                                tf.reshape(d.mean(), -1)], axis=0), 
                               scale_diag = tf.concat([tf.reshape(a.stddev(), -1),
                                                       tf.reshape(b.stddev(), -1),
                                                       tf.reshape(c.stddev(), -1),
                                                       tf.reshape(d.stddev(), -1)], axis=0)
                          )

        p = tfd.MultivariateNormalDiag(loc = tf.concat([tf.reshape(e.mean(), -1), 
                                                tf.reshape(f.mean(), -1), 
                                                tf.reshape(g.mean(), -1), 
                                                tf.reshape(h.mean(), -1)], axis=0), 
                               scale_diag = tf.concat([tf.reshape(e.stddev(), -1),
                                                       tf.reshape(f.stddev(), -1),
                                                       tf.reshape(g.stddev(), -1),
                                                       tf.reshape(h.stddev(), -1)], axis=0)
                          )

        self.kl_d = tfp.distributions.kl_divergence(p,q)

    def __call__(self, inputs, training=None, teacher_forcing = False):

        predictions = []
        x, state = self.rnn(inputs[:, :self.window_size, :])
        x = self.dense_var(x, first=True)
        predictions.append(x)

        for d in range(self.gamma-1):
            if teacher_forcing:
                x = inputs[:, self.window_size+d, :]
            else:
                x = self.DistributionLambda(x).mean()
                if np.logical_and(self.n_op > 1, d<self.lag):
                    x = tf.concat([inputs[:, self.window_size+d, :-1], x[:, -1:]], 1)
            
            x, state = self.rnn_cell(x, states=state, training=training)
            x = self.dense_var(x, first=False if self.sampling == 'once' else True)
            predictions.append(x)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        p1 = self.DistributionLambda(predictions)
        
#         if training:
        self.get_kl
        return p1

    # @tf.function
    def predict(self, x, n_steps=3, teacher_forcing = False, batch_size=None, verbose=False, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        pred = [self(x) for _ in range(n_steps)]

        means = tf.convert_to_tensor([p.mean() for p in pred])
        vars = tf.convert_to_tensor([p.variance() for p in pred])

        mean = tf.reduce_mean(means, 0)
        model_var = tf.math.reduce_variance(means, 0)
        data_var = tf.reduce_mean(vars, 0)

        std = tf.math.sqrt(model_var + data_var)
        return mean.numpy(), std.numpy(), {'model':tf.math.sqrt(model_var).numpy(), 'data':tf.math.sqrt(data_var).numpy()}

