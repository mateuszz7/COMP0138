# ONLY WORKS ON WINDOWS
"""
This file is based of the file michael_morris_github_code/Optimisation - re-using, modifying and adding to this file.
This file also makes calls to functions in the michael_morris_github_code folder.
"""

import sys
sys.path.append("./../michael_morris_github_code/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import Metrics
from optimiser_tools import *
from DataConstructor import *

from NNs import *
from bayes_opt import BayesianOptimization
from Test_Fn import Test_fn

import evidential_deep_learning as edl


# Class to evaluate a set of hyper parameters. 
# Using a class allows the __call__ function to be used for different models with different configurations
# class is used wih bayesian optimization to find the best hyper parameters
class Eval_Fn:
    def __init__(self, data_type, filename='generated_1', root='', model=None, n_folds = 5, season = 2015, gamma=28, plot=True, verbose=True, n_queries=49, min_score=-25, batch_size=32, **kwargs):
        self.model = model  
        self.n_folds = n_folds      # number of folda (k fold cross validation)
        self.season = season
        self.gamma = gamma      
        self.plot = plot            # save plots validation set forecasts  
        self.verbose = verbose      # print during training
        self.min_score = min_score  # score to give hyper paremeters if they break and get -infinity
        self.root = root            # save directory
        self.batch_size = batch_size

        # get data for training, text data unused.
        self._data = DataConstructor(test_season=self.season, gamma=self.gamma, **kwargs)
        self.x_train, self.y_train, self.x_test, self.y_test = self._data(data_type, filename=filename)


    def __call__(self, **kwargs):
        score = {}
        plt.clf()
        print('HERE0', kwargs)

        for fold in range(self.n_folds):
            #print('IN FOLD {}'.format(fold))
            # try:
                # split data into train and validation folds
                if isinstance(self.x_train, list):
                    x_val = [d[-(365*(fold+1)): -(365*(fold)+1)] for d in self.x_train]
                    x_tr = [d[:-(365*(fold+1))] for d in self.x_train]

                else:
                    x_val = self.x_train[-(365*(fold+1)): -(365*(fold)+1)]
                    x_tr = self.x_train[:-(365*(fold+1))]

                y_val = self.y_train[-(365*(fold+1)): -(365*(fold)+1)]
                y_tr = self.y_train[:-(365*(fold+1))]

                val_dates = self._data.train_dates[-365*(fold+1): -(365*fold)-1]
                train_dates = self._data.train_dates[:-365*(fold+1)]

                _model = self.model(n_batches=int(len(y_tr)/self.batch_size),**kwargs)

                # define loss, epochs learning rate
                if _model.loss == 'NLL':
                    def loss(y, p_y):
                        return -p_y.log_prob(y)
                if _model.loss == 'MSE':
                    loss = tf.keras.losses.mean_squared_error
                if _model.loss == 'EDL':
                    loss = edl.losses.EvidentialRegression

                if 'epochs' in kwargs:
                    epochs = int(kwargs['epochs'])
                else:
                    epochs = self.epochs

                if 'lr_power' in kwargs:
                    lr = np.power(10, kwargs['lr_power'])
                else:
                    lr = 1e-3

                # compile and train model
                _model.compile(loss=loss,
                               optimizer=tf.optimizers.Adam(learning_rate=lr),
                               )
                _model.fit(x_tr, y_tr, epochs=epochs, batch_size=32, verbose=self.verbose)

                # make forecasts for validation set, rescale to same scale as ILI rate (not 0 - 1)
                mean, std, model, data = _model.predict(x_val, verbose=self.verbose)
                predictions = (mean, std, {'model': model, 'data': data})
                y_val = y_val._numpy()
                try: 
                    df = convert_to_df(predictions, y_val, val_dates + dt.timedelta(days = self.gamma), self._data, type=_model.forecast_type)
                except Exception as e:
                    try:
                        predictions = (predictions[0]._numpy(), predictions[1], {'Model_Uncertainty': predictions[2]['Model_Uncertainty'], 'Data_Uncertainty': predictions[2]['Data_Uncertainty']})
                    except:
                        predictions = (predictions[0], predictions[1], {'Model_Uncertainty': predictions[2]['Model_Uncertainty'], 'Data_Uncertainty': predictions[2]['Data_Uncertainty']})
                    try:
                        df = convert_to_df(predictions, y_val, val_dates + dt.timedelta(days = self.gamma), self._data, type=_model.forecast_type)
                    except Exception as e:
                        predictions[1][np.isinf(predictions[1])] = 10000
                        predictions[2]['Model_Uncertainty'][np.isinf(predictions[2]['Model_Uncertainty'])] = 5000
                        predictions[2]['Data_Uncertainty'][np.isinf(predictions[2]['Data_Uncertainty'])] = 5000

                        print('AFTER REPLACE')
                        print('preds0', np.any(np.isinf(predictions[0])), np.any(np.isneginf(predictions[0])), type(predictions[0]))
                        print('preds1', np.any(np.isinf(predictions[1])), np.any(np.isneginf(predictions[1])), type(predictions[1]))
                        print('predsMUNC', np.any(np.isinf(predictions[2]['Model_Uncertainty'])), np.any(np.isneginf(predictions[2]['Model_Uncertainty'])), type(predictions[2]['Model_Uncertainty']))
                        print('predsDUNC', np.any(np.isinf(predictions[2]['Data_Uncertainty'])), np.any(np.isneginf(predictions[2]['Data_Uncertainty'])), type(predictions[2]['Data_Uncertainty']))

                        df = convert_to_df(predictions, y_val, val_dates + dt.timedelta(days = self.gamma), self._data, type=_model.forecast_type)


                # get score for fold, 2 options depending on whether the forecast is a list or a single prediction
                try:
                    score[fold] = Metrics.nll(df[self.gamma])
                except:
                    score[fold] = np.sum(np.asarray([Metrics.nll(d) for d in df.values()]))

                # can be useful to plot the validation curves to check things are working 
                if self.plot:
                    for idx, d in enumerate(df.values()):
                        plt.subplot(len(df.keys()), 1, idx+1)
                        plt.plot(d.index, d['True'], color='black')
                        plt.plot(d.index, d['Pred'], color='red')
                        plt.fill_between(d.index, d['Pred']+d['Std'], d['Pred']-d['Std'], color='red', alpha=0.3)
                        plt.show()

        if self.plot:
            if not os.path.exists(self.root):
                os.mkdir(self.root)
            figs = os.listdir(self.root)
            nums=[-1]
            for f in figs:
                if 'fig' in f:
                    nums.append(int(f.split('_')[1].split('.')[0]))

            plt.savefig(self.root+'fig_'+str(max(nums)+1)+'.pdf')

        try:
            # NLL can be give nan values, try to prevent this breaking things
            if np.isfinite(-sum(score.values())):
                return -sum(score.values())
            else:
                return self.min_score
        except:
            return self.min_score

def run_bayesian_optimisation(root, data_type, filename='generated_1', iterations=20, verbose=2, pbounds=None, model=FIB_RNN):
    model = model
    if not pbounds:
        pbounds = model.pbounds
    gamma = 28

    n_folds = 5 # increase this to improve rubustness, will get slower

    eval = Eval_Fn(data_type, filename=filename, model=model, root = root, gamma=gamma, plot=False, n_folds=n_folds, verbose=False)
    optimizer = BayesianOptimization(
        f=eval,
        pbounds=pbounds,
        random_state=1,
        verbose=verbose
    )

    optimizer = load_steps(root, optimizer)

    optimizer.maximize(
        init_points=5,
        n_iter=iterations)

    save_steps(root, optimizer)

    # Test_fn(root = root, model = model, gammas=[gamma], test_seasons = [2015])