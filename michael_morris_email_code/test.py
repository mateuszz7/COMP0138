import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import Metrics

from IRNN_Bayes import *
from scipy.stats import norm

from train_functions import fit
from DataConstructor import *
import numpy as np
import tqdm
import datetime as dt
import pandas as pd
import time
from optimiser_tools import *

tfd = tfp.distributions

root = 'Results/IRNN_Bayes/'
params = load_best(root)['best']['params']

window_size = 54
batch_size = 32
country = 'US'
lag = {'US':14, 'UK':7}[country]
test_season = 2015


epochs = int(params['epochs'])
kl_power = params['kl_power']
lr_power = params['lr_power']
n_op = int(params['n_op'])
op_scale_pwr = params['op_scale_pwr']
p_scale_pwr = params['p_scale_pwr']
q_scale_pwr = params['q_scale_pwr']
rnn_units = int(params['rnn_units'])




# rnn_units = 44
# op_scale_pwr = -1.5
p_scale_pwr = -2.5
# q_scale_pwr = -2.4
kl_power = -3.2
# lr_power = -3.3
# n_op = 50
gamma = 28
no_ili_input = True

for test_season in [2015, 2016, 2017, 2018]:
    for run_num in range(10):
        _data = DataConstructor(test_season = test_season, country=country, full_year=False, gamma = gamma, window_size = 54, teacher_forcing=True, n_queries = n_op-1)
        x_train, y_train, x_test, y_test = _data()

        if no_ili_input:
            x_train[:, :, -1] = 0
            x_test[:, :, -1] = 0

        x_train = tf.cast(x_train, tf.float32)[:,:,-n_op:]
        y_train = tf.cast(y_train, tf.float32)[:,:,-n_op:]
        x_test = tf.cast(x_test, tf.float32)[:,:,-n_op:]
        y_test = tf.cast(y_test, tf.float32)[:,:,-n_op:]



        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        def loss_fn(y, p_y):
            return -p_y.log_prob(y)

        optimizer = tf.optimizers.Adam(learning_rate=10**lr_power)

        n_samples = 5
        prediction_steps= 3

        # spent ages trying to speed up training, project = 0 and speedy_training = False is best.
        _model = IRNN_Bayes(kl_power=kl_power, 
                            n_op=n_op,
                            op_scale_pwr=op_scale_pwr,
                            p_scale_pwr=p_scale_pwr,
                            q_scale_pwr=q_scale_pwr,
                            rnn_units = rnn_units, 
                            gamma=gamma,       
                            window_size=54, 
                            project=0, 
                            lag = lag,
                            n_samples=n_samples,
                            )

        pred = _model(x_test)
        _model, history = fit(_model, 
                                train_dataset,
                                optimizer=optimizer, 
                                epochs = epochs, 
                                loss_fn = loss_fn,  
                                prediction_steps = prediction_steps,
                                speedy_training=False
                                )


        y_pred = _model.predict(x_test, 25, verbose=True)

        mean_all = y_pred[0]
        std_all = y_pred[1]
        data_all = y_pred[2]['data']
        model_all = y_pred[2]['model']

        new_dates =  {2015: pd.date_range(dt.date(2015, 10, 19), dt.date(2016,5, 14)),
                    2016: pd.date_range(dt.date(2016, 10, 17), dt.date(2017,5, 13)),
                    2017: pd.date_range(dt.date(2017, 10, 18), dt.date(2018,5, 12)),
                    2018: pd.date_range(dt.date(2017, 10, 20), dt.date(2018,5, 11))}
        date_mask = [date in new_dates[test_season] for date in _data.test_dates]

        for g in range(28):
            true = _data.ili_scaler.inverse_transform(y_test[:, g, -1:]).squeeze()
            mean = _data.ili_scaler.inverse_transform(mean_all[:, g, -1:]).squeeze()
            std = _data.ili_scaler.inverse_transform((mean_all+std_all)[:, g, -1:]).squeeze() - mean
            data = _data.ili_scaler.inverse_transform((mean_all+data_all)[:, g, -1:]).squeeze() - mean
            model = _data.ili_scaler.inverse_transform((mean_all+model_all)[:, g, -1:]).squeeze() - mean

            df = pd.DataFrame(columns = ['True', 'Pred', 'Std', 'Data', 'Model'], 
                                data = np.asarray([true, mean, std, data, model]).T)
            df.to_csv('Results/No_ILI_input/' + country+'_'+str(run_num) + '_Forecasts_100_'+str(test_season) + '-' + str(test_season-1999) + str(g+1) + 'days_ahead'+'.csv')

            if g in [6,13,20,27]:
                plt.subplot(2,2,int((g+1)/7))
                plt.plot(_data.test_dates, true, color='black',label="$\gamma = $" + str(g+1))
                plt.plot(_data.test_dates, mean,color='red')
                print(Metrics.skill(df))
                # for n in [0.67, 1.96]:
                for n in [1]:
                    plt.fill_between(_data.test_dates,
                                    mean+n*std,
                                    mean-n*std,
                                    color='red',
                                    linewidth=0,
                                    alpha=0.3)
                    plt.plot(_data.test_dates, mean+n*data, color='blue', linewidth=1, alpha=0.5, label = '$\sigma_d$')
                    plt.plot(_data.test_dates, mean-n*data, color='blue', linewidth=1, alpha=0.5)
                    plt.plot(_data.test_dates, mean+n*model, color='green', linewidth=1, alpha=0.5, label = '$\sigma_m$')
                    plt.plot(_data.test_dates, mean-n*model, color='green', linewidth=1, alpha=0.5)
                plt.legend()
                # plt.ylim([0.5,4])
                plt.tight_layout()
            plt.show()
            plt.savefig('Results/No_ILI_input/' + country+'_'+str(run_num) + '_Forecasts_'+str(test_season) + '-' + str(test_season-1999) + '.pdf')
        plt.clf()