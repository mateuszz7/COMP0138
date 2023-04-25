import pandas as pd
from scipy import stats
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import json
import datetime as dt
import matplotlib.pyplot as plt
import copy
import jsonlines
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.simplefilter("ignore")

class sim_score:
    def get_embedding(self, query, root=None):
        if root is None and not hasattr(self, 'root') :
            for root in ['/home/mimorris/Data/vectors_unzipped/', '/Users/michael/Documents/datasets/Search_Queries/']:
                if os.path.exists(root):
                    self.root = root

        if not hasattr(self, 'vectors'):
            self.vectors = pd.read_csv(os.path.join(self.root,'Twitter_word_embeddings_CBOW.csv'),
                                       header=None)

            f = open(os.path.join(self.root, 'vocabulary.txt'), "r")
            vocab = f.read()
            vocab = vocab.split('\n')[:-1]
            self.vectors.index = vocab

        query = query.split(' ')
        embedding = []
        for word in query:
            try:
                embedding.append(self.vectors.loc[word].values)
            except:
                embedding.append(np.zeros(self.vectors.shape[1]))
        embedding = np.asarray(embedding).mean(0)[np.newaxis, :]
        return embedding

    def get_score(self, word, pos=['flu', 'fever', 'flu', 'flu medicine', 'gp', 'hospital'],
                              neg=['bieber', 'ebola', 'wikipedia'], gamma=0.001):
        embedding = self.get_embedding(word)
        
        if not hasattr(self, 'pos'):
            self.pos = np.asarray([self.get_embedding(p) for p in pos]).squeeze()
            self.neg = np.asarray([self.get_embedding(n) for n in neg]).squeeze()

        pos = cosine_similarity(embedding.reshape(1, -1), self.pos)
        neg = cosine_similarity(embedding.reshape(1, -1), self.neg)

        pos = ((pos + 1) / 2).sum()
        neg = ((neg + 1) / 2).sum() + gamma
        return pos / neg
    
class query_selection:
    def __call__(self, queries, daily_wILI, n_queries=99, data_season=2015, country = 'UK', root='data/'):
        self.root=root

        # Similarity Scores  
        if os.path.exists(os.path.join(self.root, str(country)+'_Similarity_Scores.csv')):
            scores = pd.read_csv(os.path.join(self.root, str(country)+'_Similarity_Scores.csv'), index_col=0)
        else:
            score = sim_score()
            scores = pd.DataFrame(index=queries.columns, columns=['similarity'], data=np.asarray(
                [score.get_score(query) for query in queries]))
            scores.to_csv('data/'+country+'_Similarity_Scores.csv')

        dates = pd.date_range(str(data_season - 4) + '/8/23', str(data_season + 1) + '/8/23')

        # remove constant frequencies
        queries = queries.loc[:, queries.loc[dates].std() > 0.01]

        scores['correlation'] = pd.DataFrame(index=queries.columns,
                                             columns=['correlation'],
                                             data=[pearsonr(daily_wILI.loc[dates].squeeze(), q)[0] for q in
                                                   queries.loc[dates].values.T])

        
        scores['correlation'] = (scores['correlation'] - min(scores['correlation'])) / (max(scores['correlation']) - min(scores['correlation']))
        scores['similarity'] = (scores['similarity'] - min(scores['similarity'])) / (max(scores['similarity']) - min(scores['similarity'])) 
        
        scores['correlation'] = scores['correlation'].fillna(scores['correlation'].min())
        scores['similarity'] = scores['similarity'].fillna(scores['similarity'].min())

        scores['distance'] = np.sqrt(np.square(1 - scores / np.tile(scores.max(), (scores.shape[0], 1))).sum(1))
        scores = scores.iloc[np.argsort(scores['distance'])]

        selected_queries = scores.index[:n_queries]
        self.scores=scores

        return selected_queries

class DataConstructor():

    def __init__(self, test_season, gamma, country='UK', window_size=49, data_season = None, lag=7, n_queries=49, num_features = None, full_year=False, selection_method='distance',
                 root=None, rescale=True, teacher_forcing=False, type='avg', **kwargs):
        if data_season is None:
            data_season = test_season - 1
        test_season = int(test_season)
        data_season = int(data_season)
        gamma = int(gamma)
        window_size = int(window_size)

        self.full_year=full_year
        n_queries = int(n_queries)
        if num_features is not None:
            n_queries = int(num_features) - 1

        self.teacher_forcing = teacher_forcing
        self.test_season = test_season
        self.lag = lag
        self.gamma = gamma
        self.window_size = window_size
        self.country = country
        self.data_season = data_season
        self.n_queries = n_queries
        self.selection_method = selection_method
        self.rescale = rescale
        self.type = type

        if root is None:
            for root in ['/home/mimorris/Data/', '/Users/michael/Documents/datasets/Data/']:
                if os.path.exists(root):
                    self.root = root
        else:
            self.root=root
            
    def load(self, kind = '7day'):
        self.ili = pd.read_csv('data/ILI_rates/ILI_rates_'+str(self.country)+'_thursday_cubic_interpolation.csv', index_col=0, parse_dates=True)
        
        if kind == '7day':
            self.all_queries = pd.read_csv('data/query_data/' + str(self.country)+'_Qs_small_7day_avg.csv', index_col=0, parse_dates=True)
        else:
            self.all_queries = pd.read_csv('data/query_data/' + str(self.country)+'_Qs_small.csv', index_col=0, parse_dates=True)

        # remove duplicate index
        self.all_queries = self.all_queries[~self.all_queries.index.duplicated(keep='first')]
        self.all_queries = self.all_queries.sort_index()

        # remove punctuation
        self.all_queries = self.all_queries.rename(
            columns={query: query.replace('+', ' ').replace(',', ' ') for query in self.all_queries.columns})

        # sort queries alphabetically and remove duplicates
        self.all_queries = self.all_queries.rename(
            columns={query: ' '.join(sorted(query.split(' '))) for query in self.all_queries.columns})
        self.all_queries = self.all_queries.loc[:, ~self.all_queries.columns.duplicated()]

        # remove nan values
        self.all_queries[np.invert(self.all_queries.isna().all(1))]

        # query_dict = {}
        # for season in [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]:
        #     q = query_selection()
        #     selected_qs = q(self.all_queries, self.ili, n_queries=3000, data_season=season, country = 'US')
        #     query_dict[str(season)] = list(selected_qs)
        # with jsonlines.open('Data/' + str(self.country) + '_Selected_Queries.json', mode='w') as writer:
        #     writer.write(query_dict)
        
        file = open('data/query_data/' + str(self.country)+'_Selected_Queries.json', 'r')
        self.selected_queries = json.load(file)
        self.queries = self.all_queries[self.selected_queries[str(self.data_season)][:self.n_queries]]
        

        self.ili_scaler = MinMaxScaler()
        self.ili_scaler.fit(self.ili.values.reshape(-1, 1))
        self.ili = pd.Series(index=self.ili.index,
                                    data=self.ili_scaler.transform(self.ili.values.reshape(-1, 1)).squeeze())

        self.query_scaler = MinMaxScaler()
        self.query_scaler.fit(self.queries)
        self.queries = pd.DataFrame(index=self.queries.index, columns=self.queries.columns,
                                    data=self.query_scaler.transform(self.queries))

    def get_dates(self, leaveoneout=False):
        dates = pd.read_csv('data/Dates.csv', index_col=0).loc[self.test_season]
          
        train_start = dt.datetime.strptime(dates['train_start'], '%Y-%m-%d').date()
        test_start = dt.datetime.strptime(dates['test_start'], '%Y-%m-%d').date()
        train_end = dt.datetime.strptime(dates['train_end'], '%Y-%m-%d').date()
        test_end = dt.datetime.strptime(dates['test_end'], '%Y-%m-%d').date()
        
        self.test_dates = pd.date_range(test_start, test_end)
        if self.full_year:
            self.test_dates = pd.date_range(dt.date(self.test_season,8,23), dt.date(self.test_season+1, 8, 22))

        self.train_dates =  pd.date_range(train_start, train_end)
        

        if leaveoneout:
            d1 = pd.date_range(self.train_dates[0], self.test_dates[0]-dt.timedelta(days=1))
            d2 = pd.date_range(self.test_dates[-1]+dt.timedelta(days=1), dt.date(2019, 6, 5))
            self.train_dates = d1.append(d2)

    def __call__(self, model='feedback', forecast_type='multi', query_forecast='True', leaveoneout = False, dtype=None, look_ahead=True):
        self.load()
        self.get_dates(leaveoneout=leaveoneout)
        if model == 'feedback':
            self.queries['ili'] = self.ili
            data = {'train':{'X':[], 'Y':[]}, 'test':{'X':[], 'Y':[]}}

            for tr_te, dates in zip(['train', 'test'], [self.train_dates, self.test_dates]):
                for date in dates:

                    in_dates = pd.date_range(date-dt.timedelta(days=self.window_size-1), date + dt.timedelta(days=(self.gamma if self.teacher_forcing else self.gamma)))
                    out_dates = pd.date_range(date+dt.timedelta(days=1), date + dt.timedelta(days=self.gamma))

                    data[tr_te]['X'].append(self.queries.loc[in_dates])
                    data[tr_te]['Y'].append(self.queries.loc[out_dates])

            x_test = np.stack(data['test']['X'])
            x_train = np.stack(data['train']['X'])
            y_test = np.stack(data['test']['Y'])
            y_train = np.stack(data['train']['Y'])
            
            x_train[:, -self.lag:, -1] = 0 
            x_test[:, -self.lag:, -1] = 0 

            if query_forecast == False:
                y_train = y_train[:, :, -1:]
                y_test = y_test[:, :, -1:]
        return x_train, y_train, x_test, y_test
    
def convert_to_df(op, true, dates, _data, gamma=0, type='FF'):
    try:
        op[2]['Model_Uncertainty'] = op[2].pop('model')
        op[2]['Data_Uncertainty'] = op[2].pop('data')
    except:
        pass

    if type == 'multi':
        if len(op) == 3:
            columns=['True', 'Pred', 'Std', 'Model_Uncertainty', 'Data_Uncertainty']
        else:
            columns=['True', 'Pred', 'Std']

        gammas = list(np.linspace(7, int(7*op[0].shape[1]/7), int(op[0].shape[1]/7)).astype(int))

        res = {}
        for g in gammas:
            try:
                mean = op[0][:, g-1, -1].squeeze()
                std = op[1][:, g-1, -1].squeeze()
                data = [true[:, g-1, -1].squeeze(), mean, std]
            except:
                mean = op[0][:, g-1, -1].numpy().squeeze()
                std = op[1][:, g-1, -1].numpy().squeeze()
                data = [true[:, g-1, -1].numpy().squeeze(), mean, std]

            try:
                try:
                    model_uncertainty = op[2]['Model_Uncertainty'][:, g-1, -1].squeeze()
                    data_uncertainy = op[2]['Data_Uncertainty'][:, g-1, -1].squeeze()

                    data.append(model_uncertainty)
                    data.append(data_uncertainy)
                except:
                    model_uncertainty = op[2]['Model_Uncertainty'][:, g-1, -1].numpy().squeeze()
                    data_uncertainy = op[2]['Data_Uncertainty'][:, g-1, -1].numpy().squeeze()

                    data.append(model_uncertainty)
                    data.append(data_uncertainy)
            except:
                pass
            df = pd.DataFrame(index = dates+dt.timedelta(days=int(g)), columns = columns[:len(data)], data = np.asarray(data).T)
            df = rescale_df(df, _data.ili_scaler)
            res[g] = df

        return res

    if type == 'single':
        if len(op) == 3:
            columns=['True', 'Pred', 'Std', 'Model_Uncertainty', 'Data_Uncertainty']
        else:
            columns=['True', 'Pred', 'Std']

        mean = op[0].squeeze()
        std = op[1].squeeze()

        data = [true.squeeze(), mean, std]

        try:
            model_uncertainty = op[2]['Model_Uncertainty'].squeeze()
            data_uncertainy = op[2]['Data_Uncertainty'].squeeze()

            data.append(model_uncertainty)
            data.append(data_uncertainy)
        except:
            pass

        df = pd.DataFrame(index = dates+dt.timedelta(days=gamma), columns = columns, data = np.asarray(data).T)
        df = rescale_df(df, _data.ili_scaler)
        return {gamma:df}

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

if __name__ == '__main__':
    _data = DataConstructor(test_season=2016, country = 'UK', window_size=28, n_queries=9, gamma=28)
    x_train, y_train, x_test, y_test = _data()