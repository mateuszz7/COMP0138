"""
This file is based of the file michael_morris_github_code/Dataconstructor - re-using, modifying and adding to this file.
This file also makes calls to functions in the michael_morris_github_code folder.
"""

from logging import raiseExceptions
import os
import numpy as np
import tqdm
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import tensorflow as tf
import json

from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler

class DataConstructor():
    def __init__(self, test_season=2015, gamma=28, window_size=49, data_season = None, n_queries=49, num_features = None, selection_method='distance',
                 root='data', rescale=True, **kwargs):

        if data_season is None:
            data_season = test_season - 1
        test_season = int(test_season)
        data_season = int(data_season)
        gamma = int(gamma)
        window_size = int(window_size)

        n_queries = int(n_queries)
        if num_features is not None:
            n_queries = int(num_features) - 1

        self.test_season = test_season
        self.gamma = gamma
        self.window_size = window_size
        self.data_season = data_season
        self.n_queries = n_queries
        self.selection_method = selection_method
        self.rescale = rescale
        self.root = root
        self.generated_dir = os.path.join(self.root, 'generated')
        return

    def rescale_data(self):
        self.ili_scaler = MinMaxScaler()
        self.ili_scaler.fit(self.wILI.values.reshape(-1, 1))
        self.wILI = pd.Series(index=self.wILI.index,
                            data=self.ili_scaler.transform(self.wILI.values.reshape(-1, 1)).squeeze())
        self.daily_wILI = pd.Series(index=self.daily_wILI.index,
                                    data=self.ili_scaler.transform(self.daily_wILI.values.reshape(-1, 1)).squeeze())

    def generate_data_linear(self, filename='generated_linear', sd=0.5):
        m, c = 0.2, 0.1
        years = 11

        #2010:0, 2011:1, 2012:2, 2013:3, 2014:4, 2015:5, 2016:6, 2017:7, 2018:8
        leap_years_idx = [2,6]
        def create_seasons(years):
            mid = 183
            x_year_first_half  = np.arange(0, mid)
            x_year_second_half  = np.arange(mid, 365)
            y_year_first_half = m*x_year_first_half + c + np.random.normal(0, sd, mid)
            y_year_second_half = -m*(x_year_second_half - (2*mid)) + c + np.random.normal(0, sd, mid-1)
            seasons = np.concatenate((y_year_first_half, y_year_second_half))
            leap_x_year_first_half = np.arange(0, mid)
            leap_x_year_second_half = np.arange(mid, 366)
            for i in range(years - 1):
                leap_year = True if (i) in leap_years_idx else False
                if leap_year:
                    y_year_first_half = m*leap_x_year_first_half + c + np.random.normal(0, sd, mid)
                    seasons = np.concatenate((seasons, y_year_first_half))
                    y_year_second_half = -m*(leap_x_year_second_half - (2*mid)) + c + np.random.normal(0, sd, mid)
                    seasons = np.concatenate((seasons, y_year_second_half))
                else:
                    y_year_first_half = m*x_year_first_half + c + np.random.normal(0, sd, mid)
                    seasons = np.concatenate((seasons, y_year_first_half))
                    y_year_second_half = -m*(x_year_second_half - (2*mid)) + c + np.random.normal(0, sd, mid-1)
                    seasons = np.concatenate((seasons, y_year_second_half))
            return seasons[:, np.newaxis]

        y = create_seasons(years)
        generated_df = pd.date_range(dt.date(2009,1,1),dt.date(2019,12,31),freq='d').to_frame(index=False, name='date')
        generated_df['wILI'] = y
        if not os.path.exists(self.generated_dir):
            print('created new directory: ', self.generated_dir)
            os.mkdir(self.generated_dir)

        generated_df.to_csv(os.path.join(self.generated_dir, '{}.csv'.format(filename)), index=False)
        return filename

    def generate_data_sine_masked(self, filename='generated_sine_masked', sd=0.1, periodicity=2):
        years = 11

        #2010:0, 2011:1, 2012:2, 2013:3, 2014:4, 2015:5, 2016:6, 2017:7, 2018:8
        leap_years_idx = [2,6]
        def create_seasons(years):
            x = np.arange(0, 365)
            y = np.sin(((periodicity * (2 * np.pi)) / 365) * x) + np.random.normal(0, sd, 365)
            seasons = y
            for i in range(years - 1):
                year_length = 366 if (i) in leap_years_idx else 365
                x = np.arange(0, year_length)
                y = np.sin(((periodicity * (2 * np.pi)) / year_length) * x) + np.random.normal(0, sd, year_length)
                seasons = np.concatenate((seasons, y))
            return seasons[:, np.newaxis]

        y = create_seasons(years)

        with open(os.path.join(self.generated_dir, 'random_mask.json')) as f:
            random_mask = json.load(f)['random_mask']
        if periodicity == 12:
            days = [31,28,31,30,31,30,31,31,30,31,30,31]
        else:
            days = np.arange(0,365, round(365/periodicity)).tolist()
            if len(days) == periodicity:
                days = days[1:]
            else:
                days = days[1:-1]
            days.append(365)
        max_days = len(y)
        this_day = 0
        i = 0
        while this_day < max_days:
            next_days = days[i%periodicity]
            y[this_day:this_day+next_days] *= random_mask[i]
            this_day += next_days
            i += 1

        generated_df = pd.date_range(dt.date(2009,1,1),dt.date(2019,12,31),freq='d').to_frame(index=False, name='date')
        generated_df['wILI'] = y
        if not os.path.exists(self.generated_dir):
            print('created new directory: ', self.generated_dir)
            os.mkdir(self.generated_dir)

        generated_df.to_csv(os.path.join(self.generated_dir, '{}.csv'.format(filename)), index=False)
        return filename

    def get_ili(self):
        self.wILI = pd.read_csv(os.path.join(self.root, 'ILI_rates', 'national_flu.csv'), index_col=-1, parse_dates=True)[
                'weighted_ili']


        # get daily dates
        dates = np.asarray([self.wILI.index[0] + dt.timedelta(days=i) for i in
                            range((self.wILI.index[-1] - self.wILI.index[0]).days + 1)])

        # shift to thursday
        dates = dates + dt.timedelta(days=3)

        # interpolate weekly to daily
        x = np.linspace(0, 1, self.wILI.shape[0])
        x2 = np.linspace(0, 1, dates.shape[0])
        f = interpolate.interp1d(x, self.wILI.values.squeeze(), kind = 'cubic')

        self.daily_wILI = pd.DataFrame(index=dates, columns=['wILI'], data=f(x2)).squeeze()

        if self.rescale:
            self.rescale_data()
    
    def get_generated_ili(self, filename):
        self.wILI = pd.read_csv(os.path.join(self.generated_dir, '{}.csv'.format(filename)), parse_dates=True)
        self.wILI = self.wILI.set_index('date')
        self.daily_wILI = self.wILI.squeeze()
        self.daily_wILI.index = self.daily_wILI.index.map(np.datetime64)

        if self.rescale:
            self.rescale_data()
        
    def get_dates(self, data_type):
        if data_type == 'generated':
            test_start = {2011:dt.date(2011, 1, 1),
                        2012:dt.date(2012, 1, 1),
                        2013:dt.date(2013, 1, 1),
                        2014:dt.date(2014, 1, 1),
                        2015:dt.date(2015, 1, 1),
                        2016:dt.date(2016, 1, 1),
                        2017:dt.date(2017, 1, 1),
                        2018:dt.date(2018, 1, 1)}
            test_end = {2011:dt.date(2011, 12, 31),
                        2012:dt.date(2012, 12, 31),
                        2013:dt.date(2013, 12, 31),
                        2014:dt.date(2014, 12, 31),
                        2015:dt.date(2015, 12, 31),
                        2016:dt.date(2016, 12, 31),
                        2017:dt.date(2017, 12, 31),
                        2018:dt.date(2018, 12, 31)}

            train_start = {2011:dt.date(2010, 1, 1),
                        2012:dt.date(2010, 1, 1),
                        2013:dt.date(2010, 1, 1),
                        2014:dt.date(2010, 1, 1),
                        2015:dt.date(2010, 1, 1),
                        2016:dt.date(2010, 1, 1),
                        2017:dt.date(2010, 1, 1),
                        2018:dt.date(2010, 1, 1)}
            train_end = {2011:dt.date(2011, 12, 31),
                        2012:dt.date(2012, 12, 31),
                        2013:dt.date(2013, 12, 31),
                        2014:dt.date(2014, 12, 31),
                        2015:dt.date(2015, 12, 31),
                        2016:dt.date(2016, 12, 31),
                        2017:dt.date(2017, 12, 31),
                        2018:dt.date(2018, 12, 31)}
        else:
            test_start = {2011:dt.date(2011, 10, 28),
                        2012:dt.date(2012, 11, 4),
                        2013:dt.date(2013, 11, 3),
                        2014:dt.date(2014, 11, 2),
                        2015:dt.date(2015, 11, 1),
                        2016:dt.date(2016, 10, 30),
                        2017:dt.date(2017, 10, 29),
                        2018:dt.date(2018, 10, 28)}
            test_end = {2011:dt.date(2012, 6, 2),
                        2012:dt.date(2013, 6, 8),
                        2013:dt.date(2014, 6, 7),
                        2014:dt.date(2015, 6, 6),
                        2015:dt.date(2016, 6, 5),
                        2016:dt.date(2017, 6, 4),
                        2017:dt.date(2018, 6, 3),
                        2018:dt.date(2019, 6, 2)}

            train_start = {2011:dt.date(2004, 3, 24),
                        2012:dt.date(2004, 3, 24),
                        2013:dt.date(2004, 3, 24),
                        2014:dt.date(2004, 3, 24),
                        2015:dt.date(2004, 3, 24),
                        2016:dt.date(2004, 3, 24),
                        2017:dt.date(2004, 3, 24),
                        2018:dt.date(2004, 3, 24)}
            train_end = {2011:dt.date(2011, 8, 19),
                        2012:dt.date(2012, 8, 15),
                        2013:dt.date(2013, 8, 14),
                        2014:dt.date(2014, 8, 13),
                        2015:dt.date(2015, 8, 12),
                        2016:dt.date(2016, 8, 11),
                        2017:dt.date(2017, 8, 10),
                        2018:dt.date(2018, 8, 9)}

        self.test_start_date = test_start[self.test_season]
        self.train_dates =  pd.date_range(train_start[self.test_season], train_end[self.test_season])
        self.test_dates = pd.date_range(test_start[self.test_season], test_end[self.test_season])

    def __call__(self, data_type, filename='generated_1'):

        if data_type == 'ili':
            self.get_ili()
        elif data_type == 'generated':
            self.get_generated_ili(filename)
        else: 
            raise Exception('Incorrect data type')

        self.get_dates(data_type)
        x_train = []
        y_train = []

        x_test = []
        y_test = []
        for date in self.train_dates:
            x = pd.DataFrame(self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1):date], columns=['wILI'])
            x = x.fillna(0)

            y = pd.DataFrame(self.daily_wILI.loc[date-dt.timedelta(days=-1): date+ dt.timedelta(days=self.gamma)], columns=['wILI'])

            x_train.append(x)
            y_train.append(y)

        for date in self.test_dates:
            x = pd.DataFrame(self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1):date], columns=['wILI'])
            x = x.fillna(0)

            y = pd.DataFrame(self.daily_wILI.loc[date-dt.timedelta(days=-1): date+ dt.timedelta(days=self.gamma)], columns=['wILI'])

            x_test.append(x)
            y_test.append(y)

        x_test = tf.cast(np.stack(x_test), dtype=tf.float32)
        x_train = tf.cast(np.stack(x_train), dtype=tf.float32)
        y_test = tf.cast(np.stack(y_test), dtype=tf.float32)
        y_train = tf.cast(np.stack(y_train), dtype=tf.float32)

        return x_train, y_train, x_test, y_test

def rescale_df(df, scaler):
    std = scaler.inverse_transform((df['Pred']+df['Std']).values.reshape(-1, 1)) - scaler.inverse_transform(df['Pred'].values.reshape(-1, 1))

    try:
        model = scaler.inverse_transform((df['Pred']+df['Model_Uncertainty']).values.reshape(-1, 1)) - scaler.inverse_transform(df['Pred'].values.reshape(-1, 1))
        data = scaler.inverse_transform((df['Pred']+df['Data_Uncertainty']).values.reshape(-1, 1)) - scaler.inverse_transform(df['Pred'].values.reshape(-1, 1))
    except:
        pass

    mean = scaler.inverse_transform(df['Pred'].values.reshape(-1, 1))
    true = scaler.inverse_transform(df['True'].values.reshape(-1, 1))

    try:
        return pd.DataFrame(index=df.index, columns=df.columns, data=np.asarray([true, mean, std, model, data]).squeeze().T)
    except:
        return pd.DataFrame(index=df.index, columns=df.columns, data=np.asarray([true, mean, std]).squeeze().T)

def rescale_array(array_type, array, scaler, y=[]):
    if array_type == 'unc':
        return scaler.inverse_transform((y+array).reshape(-1, 1)) - scaler.inverse_transform(y.reshape(-1, 1)) 
    elif array_type == 'y':
        return scaler.inverse_transform(array.reshape(-1, 1)) 
    else:
        raise Exception('Incorrect array type')
