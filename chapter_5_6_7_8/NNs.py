"""
This file is based of the file michael_morris_github_code/IRNN - re-using, modifying and adding to this file.
This file also makes calls to functions in the michael_morris_github_code folder.
This file also uses GRU_cell_variational and Dense_Variational_Reparam from the michael_morris_email_code folder for the FI-ERNN.
"""

import numpy as np
from scipy.fftpack import cs_diff
import tensorflow as tf
# mac config
import logging
tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel(logging.ERROR)
import tensorflow_probability as tfp
from optimiser_tools import *
import tqdm
import evidential_deep_learning as edl

import warnings
warnings.simplefilter("ignore")

import sys
sys.path.append("./../michael_morris_email_code/")
from GRU_cell_variational import *
from Dense_Variational_Reparam import *

import multiprocessing
PROCESSORS = multiprocessing.cpu_count()
print(PROCESSORS)

tfd = tfp.distributions
class histories: pass

def loss(y, p_y):
    return -p_y.log_prob(y)

def random_gaussian_initializer(shape, dtype):
    n = int(shape / 2)
    loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
    loc = tf.Variable(
        initial_value=loc_norm(shape=(n,), dtype=dtype)
    )
    scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
    scale = tf.Variable(
        initial_value=scale_norm(shape=(n,), dtype=dtype)
    )
    return tf.concat([loc, scale], 0)

def random_gaussian_initializer2(shape, dtype):
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
    else:
        n = (shape[0], int(shape[-1]/2))
        loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
        init = tf.Variable(
            initial_value=loc_norm(shape=shape, dtype=dtype)
        )
    return init

class FIB_RNN(tf.keras.Model):

    # parameters for data builder
    model_type = 'feedback'
    forecast_type = 'multi' # does the model forecast once (single) or make a series or forecasts? (feedback)
    loss = 'NLL'
    query_forecast = 'False' # does the network forecast queries or just ILI?

    # upper and lower limits for optimization, good hyper parameters defined as standard.
    pbounds = {'rnn_units':(2,50),        # units in rnn layer
               'kl_power':(-5,0),           # KL annealing term = 10^kl_power
               'op_scale':(0.001, 0.1),      # scaling factor for output
               'prior_scale':(1e-10, 1e-1),  # prior stddev
               'epochs':(200,500),           # epochs to train for
               'lr_power':(-4, -2),         # learning rate = 10^lr_power
               'q_scale':(0.001, 0.1),       # posterior scaling factor1
               }

    def __init__(self, rnn_units=128, kl_power=-3, op_scale=0.05,
                 prior_scale=0.005, q_scale=0.02, gamma=28, n_batches=100, window_size=49, **kwargs):
        super().__init__()
        num_features = 1
        rnn_units = int(rnn_units)
        self.gamma = gamma
        self.kl_weight = np.power(10.0, kl_power)

        self.gru_cell = tf.keras.layers.GRUCell(rnn_units)
        self.gru = tf.keras.layers.RNN(self.gru_cell, return_state=True)

        def posterior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            posterior_model = tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype, trainable=True,
                                         initializer=lambda shape, dtype: random_gaussian_initializer(shape, dtype)
                                         ),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.MultivariateNormalDiag(loc=t[..., :n],
                                               scale_diag=1e-5 + q_scale*tf.nn.softplus(c + t[..., n:])),
                )),
            ])
            return posterior_model

        def prior_trainable(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = tf.keras.Sequential([
                tfp.layers.VariableLayer(n),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.MultivariateNormalDiag(loc = t, scale_diag = prior_scale*tf.ones(n))
                ))
            ])
            return prior_model

        self.dense_variational = tfp.layers.DenseVariational(units=num_features*2,
                                                             make_posterior_fn=posterior,
                                                             make_prior_fn=prior_trainable,
                                                             kl_weight = self.kl_weight/n_batches,
                                                             kl_use_exact=True
                                                             )
        c = np.log(np.expm1(1.))
        self.distribution_lambda = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :num_features],
                                 scale=1e-5 + op_scale*tf.nn.softplus(c + t[..., num_features:])
                                 )
        )
    def __call__(self, inputs, training=None):
        predictions = []
        x, *state = self.gru(inputs[:, :-1, :])
        x = self.dense_variational(x)
        predictions.append(x)

        for i in range(self.gamma-1):
            x = self.distribution_lambda(x).sample()

            x, state = self.gru_cell(x, states=state)
            x = self.dense_variational(x)
            predictions.append(x)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        predictions = self.distribution_lambda(predictions)

        return predictions

    def predict(self, x, n_steps=25, batch_size=None, verbose=False, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        pred = []
        for _ in tqdm.trange(n_steps, disable = np.invert(verbose)):
            pred.append(self(x))

        means = np.asarray([p.mean() for p in pred])
        vars = np.asarray([p.variance() for p in pred])
        mean = means.mean(0)

        std = np.sqrt(means.var(0) + vars.mean(0))

        return mean, std, np.sqrt(means.var(0)), np.sqrt(vars.mean(0))

    def predict_stable(self, x, n_steps=25):
        percent_diff = 1
        diffs=[]
        all_preds = []
        all_preds.append(self.predict(x, n_steps))
        # i = n_steps
        # print('step', i)
        while percent_diff > 0.1:
            # i += n_steps
            # print('step', i)
            this_preds = self.predict(x, n_steps)

            all_means = np.array(all_preds)[:,0].mean(axis=0)
            this_means = this_preds[0]
            percent_diff = np.absolute((this_means-all_means)/all_means).mean()
            # print(percent_diff)
            diffs.append(percent_diff)
            all_preds.append(this_preds)
        all_preds = np.array(all_preds)
        print(self.gamma, ' diffs ', diffs)
        
        return all_preds[:,0].mean(axis=0), all_preds[:,1].mean(axis=0), all_preds[:,2].mean(axis=0), all_preds[:,3].mean(axis=0)

class FI_BRNN(tf.keras.Model):

    # parameters for data builder
    model_type = 'feedback'
    forecast_type = 'multi' # does the model forecast once (single) or make a series or forecasts? (feedback)
    loss = 'NLL'             # 
    query_forecast = 'False' # does the network forecast queries or just ILI?

    # upper and lower limits for optimization, good hyper parameters defined as standard.
    pbounds = {'rnn_units':(25,125),        # units in rnn layer
               'kl_power':(-3,0),           # KL annealing term = 10^kl_power
               'op_scale':(0.01, 0.1),      # scaling factor for output
               'prior_scale':(1e-4, 1e-2),  # prior stddev
               'epochs':(100,1000),           # epochs to train for
               'lr_power':(-4, -2),         # learning rate = 10^lr_power
               'q_scale':(0.001, 0.1),       # posterior scaling factor
               }

    def __init__(self, rnn_units=128, kl_power=-3, op_scale=0.05,
                 prior_scale=0.005, q_scale=0.02, gamma=28, n_batches=100, window_size=49,
                 project=0, 
                 n_samples=3,
                 sampling='once', **kwargs):
        super().__init__()
        num_features = 1
        rnn_units = int(rnn_units)
        self.window_size = int(window_size)
        self.gamma = gamma
        self.kl_weight = np.power(10.0, kl_power)
        p_scale = np.power(10.0, prior_scale)
        self.n_op = 1
        self.sampling=sampling

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
        
        def other_prior(shape, name, initializer,  scale = None, project = False,  n_samples = n_samples, regularizer=None, constraint=None, cashing_device=None, dtype=tf.float32):
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

        self.gru_cell = GRU_Cell_Variational(
            int(rnn_units), 
            kernel_prior_fn = other_prior, 
            kernel_posterior_fn = rnn_posterior,
            recurrent_kernel_prior_fn = other_prior, 
            recurrent_kernel_posterior_fn = rnn_posterior,
            bias_posterior_fn = other_prior, 
            bias_prior_fn = rnn_posterior,
            project = project,
            n_samples = n_samples,
            scale = q_scale,
            kl_weight = self.kl_weight,
            sampling=self.sampling
            
        )
        self.gru = tf.keras.layers.RNN(self.gru_cell, return_state=True)

        self.dense = tf.keras.layers.Dense(2, activation='relu')
        c = np.log(np.expm1(1.))
        self.distribution_lambda = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :num_features],
                                 scale=1e-5 + op_scale*tf.nn.softplus(c + t[..., num_features:])
                                 )
        )

    def __call__(self, inputs, training=None):
        predictions = []
        x, *state = self.gru(inputs[:, :self.window_size, :])
        x = self.dense(x)
        predictions.append(x)

        for i in range(self.gamma-1):
            x = self.distribution_lambda(x).sample()

            x, state = self.gru_cell(x, states=state, training=training)
            x = self.dense(x)
            predictions.append(x)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        predictions = self.distribution_lambda(predictions)

        return predictions

    def predict(self, x, n_steps=25, batch_size=None, verbose=False, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        pred = []
        for _ in tqdm.trange(n_steps, disable = np.invert(verbose)):
            pred.append(self(x))

        means = np.asarray([p.mean() for p in pred])
        vars = np.asarray([p.variance() for p in pred])
        mean = means.mean(0)

        std = np.sqrt(means.var(0) + vars.mean(0))

        return mean, std, np.sqrt(means.var(0)), np.sqrt(vars.mean(0))

    def predict_stable(self, x, n_steps=25):
        percent_diff = 1
        diffs=[]
        all_preds = []
        all_preds.append(self.predict(x, n_steps))
        while percent_diff > 0.1:
            # i += n_steps
            # print('step', i)
            this_preds = self.predict(x, n_steps)

            all_means = np.array(all_preds)[:,0].mean(axis=0)
            this_means = this_preds[0]
            percent_diff = np.absolute((this_means-all_means)/all_means).mean()
            # print(percent_diff)
            diffs.append(percent_diff)
            all_preds.append(this_preds)
        all_preds = np.array(all_preds)
        print(self.gamma, ' diffs ', diffs)
        
        return all_preds[:,0].mean(axis=0), all_preds[:,1].mean(axis=0), all_preds[:,2].mean(axis=0), all_preds[:,3].mean(axis=0)

class FI_ERNN(tf.keras.Model):
    forecast_type = 'multi' # does the model forecast once (single) or make a series or forecasts? (feedback)
    loss = 'EDL'

    # upper and lower limits for optimization, good hyper parameters defined as standard.
    pbounds = {'rnn_units':(25,125),        # units in rnn layer
               'kl_power':(-3,0),           # KL annealing term = 10^kl_power
               'op_scale':(0.01, 0.1),      # scaling factor for output
               'prior_scale':(1e-4, 1e-2),  # prior stddev
               'epochs':(100,1000),           # epochs to train for
               'lr_power':(-4, -2),         # learning rate = 10^lr_power
               'q_scale':(0.001, 0.1)       # posterior scaling factor
               }

    def __init__(self, rnn_units=128, kl_power=-3, op_scale=0.05,
                 prior_scale=0.005, q_scale=0.02, gamma=28, n_batches=100, window_size=49, **kwargs):
        super().__init__()
        rnn_units = int(rnn_units)
        self.gamma = gamma

        self.gru_cell = tf.keras.layers.GRUCell(rnn_units)
        self.gru = tf.keras.layers.RNN(self.gru_cell, return_state=True)
        self.edl_layers = [edl.layers.DenseNormalGamma(1) for _ in range(gamma)]

    def __call__(self, inputs, training=None):
        predictions = []
        # x = tf.transpose(inputs[:, :-1, :], [1, 0, 2])
        # x, *state = self.gru(x)
        x, *state = self.gru(inputs[:, :-1, :])
        # print('HEREEE ', x.shape, x)
        # x = self.edl_layer(tf.reshape(x, [-1, 1]))
        x = self.edl_layers[0](x)
        predictions.append(x)
        x, _, _, _ = tf.split(x, 4, axis=-1)

        for i in range(1, self.gamma):
            x, state = self.gru_cell(x, states=state)
            x = self.edl_layers[i](x)
            # x = self.edl_layer(tf.reshape(x, [-1, 1]))
            predictions.append(x)
            x, _, _, _ = tf.split(x, 4, axis=-1)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        # print('HEREE', predictions.shape)
        return predictions

    def predict(self, x, n_steps=1, batch_size=None, verbose=False, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        mu, v, alpha, beta = tf.split(self(x), 4, axis=-1)

        mean = mu
        data_unc = beta / (v * (alpha - 1))
        model_unc = beta / (alpha - 1)
        std = np.sqrt(data_unc + model_unc)

        return mean, std, np.sqrt(model_unc), np.sqrt(data_unc)

    def predict_stable(self, x):
        return self.predict(x)

class FI_NNs():
    def __init__(self, num_irnns=10, rnn_units=128, kl_power=-3, op_scale=0.05,
                prior_scale=0.005, q_scale=0.02, gamma=28, n_batches=100,
                lr_power=3, irnn_type = FIB_RNN, window_size=49, edl_coeff=1, **kwargs):
        self.num_irnns = num_irnns
        model = irnn_type
        self.irnns = [model(rnn_units, kl_power, op_scale,
                 prior_scale, q_scale, gamma, n_batches, window_size, **kwargs) for _ in range(num_irnns)]
        lr=np.power(10, lr_power)
        use_loss = loss
        if model.loss == 'EDL':
            def hyperparam_edl_loss(true, pred):
                return edl.losses.EvidentialRegression(true, pred, coeff=edl_coeff)
            use_loss = hyperparam_edl_loss
        for irnn in self.irnns:
            irnn.compile(loss=use_loss, optimizer=tf.optimizers.Adam(learning_rate=lr))

    def train(self, x_train, y_train, epochs=10, batch_size=32, verbose=False, checkpoints=False):
        def run_train(i):
            if checkpoints:
                checkpoint_dir = '../../models/IRNNs{}_checkpoints'.format(i)
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir,
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
                losses = self.irnns[i].fit(x_train, y_train, epochs=epochs,batch_size=batch_size,
                verbose=verbose, callbacks=[checkpoint_callback]).history['loss'][-1]
                self.irnns[i].load_weights(checkpoint_dir)
                return losses
            else:
                return self.irnns[i].fit(x_train, y_train, epochs=epochs,batch_size=batch_size,
                verbose=verbose).history['loss'][-1]

        losses = []
        with multiprocessing.pool.ThreadPool(PROCESSORS) as pool:
            losses = pool.map(run_train, range(len(self.irnns)))

        return losses

    def get_predictions(self, x_test, stable=False):
        def run_predictions(irnn):
            return irnn.predict(x_test)

        def run_predictions_stable(irnn):
            return irnn.predict_stable(x_test)

        pred_func = run_predictions_stable if stable else run_predictions

        predictions = []
        print('LEN', len(self.irnns))
        with multiprocessing.pool.ThreadPool(PROCESSORS) as pool:
            predictions = pool.map(pred_func, [irnn for irnn in self.irnns])
        predictions = np.array(predictions)
        
        return predictions[:,0].mean(axis=0), predictions[:,1].mean(axis=0), predictions[:,2].mean(axis=0), predictions[:,3].mean(axis=0)
