from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from graphviz import Source
import pydotplus

import os
import string
import random
from random import shuffle

# initialize the database with None values
database = {
    'raw_data': {
        'df_train': None,
        'df_test': None,
        'y_train': None,
        'y_test': None,
        'headers': None
    },
    'preprocessed_data': {
        'df_train': None,
        'df_test': None,
        'y_train': None,
        'y_test': None,
        'headers': None
    },
    # all the data from the training step
    'training_data': {
        'real_training_score': None,
        'DecisionTreeRegressor': None,
        'max_features': None,
        'min_samples_leaf': None,
        'max_depth': None
    },
    # the score before actual training (only shown in preprocessing steps)
    'training_score': None,
    # the score in testing
    'real_test_score': None,
    # the last sample and predicted value
    'sample': None,
    'prediction': None,
    # the last set hashes for the .png in training and deploying
    'dtree_hash': None,
    'dtree_path_hash': None,
    # for knowing the categories and equivalent codes in the trained model
    'mapping': None
}


# I attempted to create categorical features here, complicates a lot
# the data in this dictionary will be a possible layout for the generated X
# The keys MUST be the name of the feature


class DataGenerator(object):

    def __init__(self, config=None, n_samples=1000):

        if config is None:
            config = self.get_default_config()

        self.config = config

        self.features_relevant = config["features_relevant"]

        self.features_noise = config["features_noise"]

        self.target = config["target"]

        self.reset()

    def reset(self):
        self.raw_data = None
        self.preprocessed_data = None

    def get_default_config(self):
        config = {

            "samples": 1000,

            "features_relevant": {
                'Anzahl der Kavitäten': ("c", (1, 2, 3)),
                'Form der Kavitäten': (
                    "c", ('A31C', 'A32B', 'A34', 'B42', 'B3', 'C')),

                'Material': ("c", ('PU', 'PE', 'PVC', 'PUT')),
                'Entformungskonzept': ("c", ('A', 'B')),
                'Schieberanzahl': ("c", (1, 2, 3, 4, 5, 6, 7)),
                'Kanaltyp': ("c", ('Kaltkanal', 'Heißkanal')),

                'Größe der Kavitäten': ("n", (1, 7)),
                'Abmaße Werkzeug': ("n", (4, 31)),
                'x': ("n", (10, 40)),
                'y': ("n", (10, 40)),
                'z': ("n", (10, 40)),
                'Temp': ("n", (170, 250)),
                'Time': ("n", (50, 17000)),
            },

            "features_noise": {
                'noise': ("n", (10, 40)),
            },

            "target": ('Kosten', (100, 2000))
        }

        return config

    def get_raw_data(self):
        if self.raw_data is None:
            self.generate_raw_data()

        return self.raw_data

    def get_preprocessed_data(self):
        if self.preprocessed_data is None:
            self.generate_preprocessed_data()

        return self.preprocessed_data

    def generate_raw_data(self):
        """
        This method creates the regression dataset with (effective rank) number of singular vectors.
        Some values in some columns will be set to NaN.
        """

        # first reset the database and picture subfolder
        # reset()

        X, y = make_regression(n_samples=self.config["samples"],
                               n_features=len(self.features_relevant),
                               n_informative=len(self.features_relevant),
                               noise=0.2)

        y = y.reshape(-1, 1)

        if len(self.features_noise):
            X_noise = np.random.rand(X.shape[0], len(self.features_noise))

            X = np.concatenate((X, X_noise), axis=1)

        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y)

        # configure features

        feature_names = list(self.features_relevant.keys()) + list(
            self.features_noise.keys())

        df_X = pd.DataFrame(X, columns=feature_names)

        for feature in feature_names:
            if feature in self.features_relevant:
                feature_data = self.features_relevant[feature]
            elif feature in self.features_noise:
                feature_data = self.features_noise[feature]
            else:
                raise ValueError(f"Feature {feature} not found")

            feature_type, feature_config = feature_data

            if feature_type == "n":
                if len(feature_config) != 2:
                    raise ValueError(
                        f"Numerical features must have config of length 2 [{feature}]")

                df_X[feature] *= feature_config[1] - feature_config[0]
                df_X[feature] += feature_config[0]

            elif feature_type == "c":

                amount_categories = len(feature_config)

                df_X[feature] = pd.cut(df_X[feature], amount_categories)

                df_X[feature].cat.categories = feature_config

            else:
                raise ValueError(
                    f"Type {feature_type} for Feature {feature} unknown")

        # configure target

        target_name, target_config = self.target

        df_y = pd.DataFrame(y, columns=[target_name])

        df_y *= target_config[1] - target_config[0]
        df_y += target_config[0]

        # Introduce NaN

        nan_mask = np.random.random(df_X.shape) < .1

        df_X = df_X.mask(nan_mask)

        self.raw_data = df_X, df_y

    def generate_preprocessed_data(self):
        if self.raw_data is None:
            self.generate_raw_data()

        df_X, df_y = self.raw_data

        df_X.dropna(inplace=True)

        df_y = df_y.loc[df_X.index]

        self.preprocessed_data = df_X, df_y


if __name__ == "__main__":
    generator = DataGenerator()

    X_df, y_df = generator.get_preprocessed_data()

    print(X_df.head())

#
#
# def reset():
#     """
#     Method will delete all instances in the database and all .png files in the picture subfolder.
#     """
#     global database, train_id
#     # reset all values in the database to None
#     for key in database.keys():
#         if isinstance(database[key], dict):
#             for inner_key in database[key].keys():
#                 database[key][inner_key] = None
#         else:
#             database[key] = None
#
#     # also delete all files in the /static/pictures folder
#     filelist = [f for f in os.listdir('static/pictures/') if f.endswith(".png")]
#     for f in filelist:
#         os.remove(os.path.join('static/pictures/', f))
#
#     # todo remove
#     # and reset the ids to 0
#     train_id = 0
#     deploy_id = 0
#
#     return
#
#
# def get_data(kind, unpack=True):
#     """
#     Read the database and return the datasets inside.
#     """
#     global database
#
#     if kind == 'training_data':
#         data = database['training_data']
#         real_training_score = data['real_training_score']
#         regressor = data['DecisionTreeRegressor']
#         max_features = data['max_features']
#         min_samples_leaf = data['min_samples_leaf']
#         max_depth = data['max_depth']
#
#         return [real_training_score, regressor, max_features, min_samples_leaf,
#                 max_depth]
#
#     if kind not in ['raw_data', 'preprocessed_data']:
#         raise Exception(
#             'This param kind is not known. (Does not point to any data)')
#
#     data = database[kind]
#     df_train = data['df_train']
#     df_test = data['df_test']
#     y_train = data['y_train']
#     y_test = data['y_test']
#     headers = data['headers']
#
#     # Return copies of the variables in the database!
#     if df_train is not None:
#         df_train, df_test, y_train, y_test, headers = df_train.copy(), df_test.copy(), \
#                                                       y_train.copy(), y_test.copy(), headers.copy()
#
#     if unpack is True and df_train is not None:
#         df_train, df_test = df_train.values, df_test.values
#
#     return [df_train, df_test, y_train, y_test, headers]
#
#
# def get_scores():
#     """
#     Returns the scores from the database.
#     :return: Dictionary with the relevant scores.
#     """
#     return {
#         'training_score': database['training_score'],
#         'real_training_score': database['training_data']['real_training_score'],
#         'real_test_score': database['real_test_score']
#     }
#
#
# def get_table(kind):
#     """
#     This method will return the table (np array) of training data including the headers (names of the features).
#     Used for displaying the data.
#     """
#     X_train, X_test, y_train, y_test, headers = get_data(kind)
#
#     full_headers = ['Index']
#     full_headers.extend(headers)
#
#     index = np.arange(len(X_train), dtype=int).reshape(len(X_train), 1)
#     # convert from list to vertical array
#     y_tr = np.array(y_train).reshape(len(y_train), 1)
#
#     table = np.hstack((np.hstack((index, y_tr)), X_train))
#
#     return [table, full_headers]
#
#
# def numericalize(df):
#     """
#     Convert all the columns with categorical data to numeric data by exchanging the alphanumeric categories with their
#     numeric codes.
#     :param df: pandas dataframe
#     :return: numpy array
#     """
#     global database
#
#     cat_columns = df.select_dtypes(['category']).columns
#
#     # save a dict with the codes and equivalent categories for each cat feature
#     mapping = {}
#     for cat_column in cat_columns:
#         mapping[cat_column] = dict(enumerate(df[cat_column].cat.categories))
#
#     # save mapping to database
#     database['mapping'] = mapping
#
#     df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
#
#     return df
#
#
# def impute(df, y, strategy, headers=None):
#     """
#     Perform operations on the data such as dropping rows/columns with missing values or imputation of NaNs.
#     """
#     if strategy not in ['drop_row', 'drop_col', 'mean', 'median']:
#         raise Exception(f'Given param strategy -> {strategy} <- not known.')
#
#     if strategy == 'drop_row':
#         # also need to drop some labels in y, thus stack X and y
#         # first convert y from list to vertical np array
#         # y = np.array(y).reshape(len(y), 1)
#         df['label'] = y
#         df.dropna(axis=0, inplace=True)
#         y = df['label']
#         df = df.drop('label', axis=1)
#
#         return df, y, headers
#
#     if strategy == 'drop_col':
#         if headers is not None:
#             seen_cols = []
#             to_pop = []
#             # remove column with NaNs from headers list
#             for (row, col), value in np.ndenumerate(df):
#                 # to not drop ints. pandas gotcha
#                 if np.isnan(value) and col not in seen_cols and not isinstance(
#                         df.iloc[row, col], int):
#                     to_pop.append(col + 1)
#                     seen_cols.append(col)
#             # sort cols (larger numbers at the end) and reverse list to pop those cols first
#             to_pop.sort()
#             to_pop.reverse()
#             for header in to_pop:
#                 headers.pop(header)
#
#         df.dropna(axis=1, inplace=True)
#
#         return df, y, headers
#
#     imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
#     df = imp.fit_transform(df)
#
#     return df, y, headers
#
#
# def build_tree(X_train, y_train, max_features=None, min_samples_leaf=1,
#                max_depth=None):
#     """
#     Build a DecisionTreeRegressor and return the tree plus its score on the training set.
#     """
#     if X_train.dtype not in [np.float64, np.int32]:
#         raise Exception('Input array X_train is not all numerical')
#
#     m = DecisionTreeRegressor(max_features=max_features,
#                               min_samples_leaf=min_samples_leaf,
#                               max_depth=max_depth)
#
#     m.fit(X_train, y_train)
#     score = m.score(X_train, y_train)
#
#     return m, score
#
#
# def preprocess(strategy):
#     """
#     Will simulate the entire preprocessing for one chosen strategy.
#     :param strategy: must either be 'raw_data' or 'processed_data'
#     """
#     df_train, df_test, y_train, y_test, headers = get_data('raw_data',
#                                                            unpack=False)
#
#     # convert categorical data to their numeric codes for further processing
#     df_train = numericalize(df_train)
#     df_test = numericalize(df_test)
#
#     # !!! only process headers once (otherwise certain indexes in list will be dropped again) !!!
#     df_train, y_train, headers = impute(df_train, y_train, strategy, headers)
#
#     df_test, y_test, _ = impute(df_test, y_test, strategy)
#
#     # something returns an array already
#     if not isinstance(df_train, (np.ndarray, np.generic)):
#         X_train, X_test = df_train.values, df_test.values
#     # if is np array
#     else:
#         X_train, X_test = df_train, df_test
#         df_train, df_test = pd.DataFrame(df_train), pd.DataFrame(df_test)
#
#     # try the preprocessed training data with a tree
#     m, score = build_tree(X_train, y_train)
#
#     # save the data to the database as preprocessed data
#     for set, path in [(df_train, 'df_train'), (df_test, 'df_test'),
#                       (y_train, 'y_train'), (y_test, 'y_test'),
#                       (headers, 'headers')]:
#         database['preprocessed_data'][path] = set
#
#     database['training_score'] = score
#
#     return
#
#
# def training(max_features, min_samples_leaf, max_depth):
#     """
#     The whole training procedure is done in this method.
#     """
#     global database
#     filename = get_hash()
#     database['dtree_hash'] = filename
#
#     X_train, X_test, y_train, y_test, headers = get_data('preprocessed_data')
#
#     # remove label from headers
#     headers.pop(0)
#
#     m, real_training_score = build_tree(X_train, y_train,
#                                         max_features=max_features,
#                                         min_samples_leaf=min_samples_leaf,
#                                         max_depth=max_depth)
#
#     real_test_score = m.score(X_test, y_test)
#
#     graph = Source(
#         export_graphviz(m, out_file=None, feature_names=headers, filled=True,
#                         rounded=True,
#                         special_characters=True, precision=3))
#     png_bytes = graph.pipe(format='png')
#     with open(f'static/pictures/{filename}.png', 'wb') as file:
#         file.write(png_bytes)
#
#     # save the model
#     database['training_data']['DecisionTreeRegressor'] = m
#     database['training_data']['real_training_score'] = real_training_score
#     database['training_data']['max_features'] = max_features
#     database['training_data']['min_samples_leaf'] = min_samples_leaf
#     database['training_data']['max_depth'] = max_depth
#
#     database['real_test_score'] = real_test_score
#
#     return real_training_score, real_test_score
#
#
# def get_train_filename():
#     """
#     Return the full filename for the latest .png picture of a dtree
#     """
#
#     return f"/static/pictures/{database['dtree_hash']}.png"
#
#
# def get_data_structure():
#     """
#     Find out about the maximum and minimum values for each feature.
#     """
#
#     [X_train, X_test, y_train, y_test, headers] = get_data('preprocessed_data')
#
#     # slice off the label
#     feature_names = headers[1:]
#
#     X = np.vstack((X_train, X_test))
#
#     mins = list(np.nanmin(X, axis=0))
#     maxs = list(np.nanmax(X, axis=0))
#
#     steps = [abs((maxs[i] - mins[i]) / 100) for i in range(len(mins))]
#
#     if len(mins) != len(maxs) or len(maxs) != len(feature_names):
#         raise Exception(
#             f'Length of mins, {len(mins)} is not equal to length of maxs, '
#             f'{len(maxs)} or length of feature names, {len(feature_names)}')
#
#     return [[feature_names[i], mins[i], maxs[i], steps[i]] for i in
#             range(len(feature_names))]
#
#
# def get_mapping():
#     return database['mapping']
#
#
# def get_slider_config():
#     """
#     Create a dictionary which can be looped through in the html template to produce a dynamic number of sliders
#     for the deployment section.
#     """
#     a = get_data_structure()
#
#     config = []
#
#     for [header, min_, max_, step] in a:
#         is_cat = False
#         codes = None
#         if isinstance(features[header], Categorical):
#             codes = get_mapping()[header]
#             is_cat = True
#             step = 1
#         slider = {
#             'name': header,
#             'is_cat': is_cat,
#             'codes': codes,
#             'slider_id': f'{header}_slider',
#             'value_id': f'{header}_value',
#             'min': min_,
#             'max': max_,
#             'step': step
#         }
#         config.append(slider)
#
#     return config
#
#
# def make_prediction(sample):
#     """
#     Given a single sample as data input return the prediction of the DecisionTreeRegressor.
#     """
#     real_training_score, regressor, max_features, min_samples_leaf, max_depth = get_data(
#         'training_data')
#
#     # reshape because of single sample
#     x = np.array(sample).reshape(1, -1)
#
#     pred = regressor.predict(x)[0]
#
#     # save the sample and prediction into database
#     database['prediction'] = pred
#     database['sample'] = sample
#
#     return pred
#
#
# def get_sample_pred():
#     """
#     For displaying sample and prediction in the deployment section.
#     """
#     feature_names = database['preprocessed_data']['headers'][1:]
#     sample = database['sample']
#     if sample is not None:
#         sample = [(feature_names[i], sample[i]) for i in
#                   range(len(feature_names))]
#     return sample, database['prediction']
#
#
# def decision_tree_path(sample, regressor, headers):
#     """
#     Save a .png image of the decision tree with a highlighted path for the single trained sample.
#     """
#
#     # make sample array 2D
#     sample = [sample]
#
#     dot_data = export_graphviz(regressor, out_file=None,
#                                feature_names=headers,
#                                filled=True, rounded=True,
#                                special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#
#     # empty all nodes, i.e.set color to white and number of samples to zero
#     # basically change encoding to design preference (white does not fit here)
#     for node in graph.get_node_list():
#         if node.get_attributes().get('label') is None:
#             continue
#         if 'samples = ' in node.get_attributes()['label']:
#             labels = node.get_attributes()['label'].split('<br/>')
#             for i, label in enumerate(labels):
#                 if label.startswith('samples = '):
#                     labels[i] = 'samples = 0'
#             node.set('label', '<br/>'.join(labels))
#             # node.set_fillcolor('white')
#
#     decision_paths = regressor.decision_path(sample)
#
#     for decision_path in decision_paths:
#         for n, node_value in enumerate(decision_path.toarray()[0]):
#             if node_value == 0:
#                 continue
#             node = graph.get_node(str(n))[0]
#             node.set_fillcolor('green')
#             labels = node.get_attributes()['label'].split('<br/>')
#             for i, label in enumerate(labels):
#                 if label.startswith('samples = '):
#                     labels[i] = 'samples = {}'.format(
#                         int(label.split('=')[1]) + 1)
#
#             node.set('label', '<br/>'.join(labels))
#
#     return graph
#
#
# def deploy(sample):
#     """
#     The deployment process.
#     """
#     global database
#     filename = get_hash()
#     database['dtree_path_hash'] = filename
#
#     # find out, if features have been dropped
#     if len(database['raw_data']['headers']) != len(
#             database['preprocessed_data']['headers']):
#         headers = get_data('preprocessed_data')[4][1:]
#     else:
#         headers = get_data('raw_data')[4][1:]
#
#     regressor = get_data('training_data')[1]
#     make_prediction(sample)
#
#     graph = decision_tree_path(sample, regressor, headers)
#     full_filename = f'static/pictures/{filename}.png'
#     graph.write_png(full_filename)
#
#     return
#
#
# def get_deploy_filename():
#     """
#     Return the full filename for the latest .png picture of a path of a sample through a dtree.
#     """
#
#     return f"/static/pictures/{database['dtree_path_hash']}.png"
#
#
# def get_hash():
#     """Generate a random string of fixed length """
#     letters = string.ascii_lowercase
#     return ''.join(random.choice(letters) for i in range(12))
