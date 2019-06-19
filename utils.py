from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import numpy as np
import pandas as pd

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


class Categorical:
    def __init__(self, categories, name):
        self.categories = categories
        self.category_count = len(categories)
        self.name = name


class Numerical:
    def __init__(self, MAX, MIN, name, value=None, ):
        self.MAX = MAX
        self.MIN = MIN
        self.value = value
        self.name = name

    def get_factor(self):
        return self.MAX - self.MIN

    def get_addend(self):
        return self.MIN


# I attempted to create categorical features here, complicates a lot
# the data in this dictionary will be a possible layout for the generated X
# The keys MUST be the name of the feature
feature_layout = {
    'Anzahl der Kavitäten': Categorical([1, 2, 3], 'Anzahl der Kavitäten'),
    'Form der Kavitäten': Categorical(['A31C', 'A32B', 'A34', 'B42', 'B3', 'C'], 'Form der Kavitäten'),
    'Größe der Kavitäten': Numerical(7, 1, 'Größe der Kavitäten'),
    'Material': Categorical(['PU', 'PE', 'PVC', 'PUT'], 'Material'),
    'Entformungskonzept': Categorical(['A', 'B'], 'Entformungskonzept'),
    'Abmaße Werkzeug': Numerical(31, 4, 'Abmaße Werkzeug'),
    'Schieberanzahl': Categorical([1, 2, 3, 4, 5, 6, 7], 'Schieberanzahl'),
    'Kanaltyp': Categorical(['Heißkanal', 'Kaltkanal'], 'Kanaltyp'),
    'x': Numerical(10, 40, 'x'),
    'y': Numerical(10, 40, 'y'),
    'z': Numerical(10, 40, 'z'),
    'Temp': Numerical(170, 250, 'Temp'),
    'Time': Numerical(50, 17000, 'Time'),

}


def get_max_features():
    return len(feature_layout)


def make_realistic(df):
    """
    Adjusts the generated datapoints to realistic values with realistic feature_names.
    df without the label!
    """
    global feature_layout

    # make_regression returns X values between -0.01 and +0.01
    # thus first create value between -1 and +1
    df.iloc[:, :] *= 100

    # no label in df!
    n_samples, n_features = df.shape

    # use n_features random features from the layout for the features of this dataset
    keys = list(feature_layout.keys())
    shuffle(keys)
    keys = keys[:n_features]

    # the keys are also the names of the features in the correct order
    feature_names = keys

    df.columns = feature_names

    # the maxs and mins for each column, find out the absolute interval of the values and divide it into sub_intervals
    # for each category of a feature
    maxs = np.nanmax(df.iloc[:, :], axis=0)
    mins = np.nanmin(df.iloc[:, :], axis=0)
    interval = abs(maxs - mins)

    num_cols, cat_cols = [], []

    for (row, col), value in np.ndenumerate(df.iloc[:, :]):
        key = keys[col]
        feature = feature_layout[key]
        # if the feature/ this column is numerical -> rescale cols afterwards all together (faster)
        # ( so just do this for row 0)
        if row == 0 and isinstance(feature, Numerical):
            factor = feature.get_factor()
            addend = feature.get_addend()
            num_cols.append([col, factor, addend])

        # if the column is categorical create category from number
        elif isinstance(feature, Categorical):
            # adding cat to list for later conversion to category type (only once for every col)
            if row == 0:
                cat_cols.append(key)

            # divide into sub_intervals and check for the sub_interval in which the points belongs
            category_count = feature.category_count
            sub_interval = interval[col] / category_count
            for cat_index in range(category_count):
                if value <= mins[col] + sub_interval * (cat_index+1):
                    # assign the right category
                    df.iloc[row, col] = feature.categories[cat_index]
                    break
                # if the cat value has not been set in the last iteration
                # (because of rounding errors in the calculation, the max value might be passing)
                elif cat_index == category_count - 1:
                    df.iloc[row, col] = feature.categories[cat_index]

    # set all the category types
    for key in cat_cols:
        df[key] = df[key].astype('category')

    # now resize all the numerical columns all together
    for col, factor, addend in num_cols:
        df.iloc[:, col] *= factor
        df.iloc[:, col] += addend

    return df, feature_names


def gen_data(n_train, n_test, n_features, effective_rank, noise):
    """
    This method creates the regression dataset with (effective rank) number of singular vectors.
    Some values in some columns will be set to NaN.
    """

    # first reset the database and picture subfolder
    reset()

    X, y, coef = make_regression(n_samples=n_train+n_test, n_features=n_features,
                                 effective_rank=effective_rank, noise=noise, coef=True)

    df = pd.DataFrame(X)
    df, feature_names = make_realistic(df)

    # decide on the columns which will hold NaNs
    col_len = df.shape[1]
    to_nan = []
    for i in range(int(col_len/3)):
        to_nan.append(np.random.randint(0, col_len))

    # set NaNs
    for (row, col), value in np.ndenumerate(df.values):
        if col in to_nan and np.random.random() < 0.1:
            df.iloc[row, col] = np.nan

    df_train, y_train = df[:n_train], y[:n_train]
    df_test, y_test = df[n_train:n_test+n_train], y[n_train:n_test+n_train]

    # create headers for the data
    headers = ['Label']
    headers.extend(feature_names)
    # for i in range(X.shape[1]):
    #     headers.append(f'Feature {i}')

    # save the data so the database as raw data
    for set, path in [(df_train, 'df_train'), (df_test, 'df_test'), (y_train, 'y_train'),
                      (y_test, 'y_test'), (headers, 'headers')]:
        database['raw_data'][path] = set


    return


def reset():
    """
    Method will delete all instances in the database and all .png files in the picture subfolder.
    """
    global database, train_id
    # reset all values in the database to None
    for key in database.keys():
        if isinstance(database[key], dict):
            for inner_key in database[key].keys():
                database[key][inner_key] = None
        else:
            database[key] = None

    # also delete all files in the /static/pictures folder
    filelist = [f for f in os.listdir('static/pictures/') if f.endswith(".png")]
    for f in filelist:
        os.remove(os.path.join('static/pictures/', f))

    # todo remove
    # and reset the ids to 0
    train_id = 0
    deploy_id = 0

    return


def get_data(kind, unpack=True):
    """
    Read the database and return the datasets inside.
    """
    global database

    if kind == 'training_data':
        data = database['training_data']
        real_training_score = data['real_training_score']
        regressor = data['DecisionTreeRegressor']
        max_features = data['max_features']
        min_samples_leaf = data['min_samples_leaf']
        max_depth = data['max_depth']

        return [real_training_score, regressor, max_features, min_samples_leaf, max_depth]

    if kind not in ['raw_data', 'preprocessed_data']:
        raise Exception('This param kind is not known. (Does not point to any data)')

    data = database[kind]
    df_train = data['df_train']
    df_test = data['df_test']
    y_train = data['y_train']
    y_test = data['y_test']
    headers = data['headers']

    # Return copies of the variables in the database!
    if df_train is not None:
        df_train, df_test, y_train, y_test, headers = df_train.copy(), df_test.copy(), \
                                                    y_train.copy(), y_test.copy(), headers.copy()

    if unpack is True and df_train is not None:
        df_train, df_test = df_train.values, df_test.values

    return [df_train, df_test, y_train, y_test, headers]


def get_scores():
    """
    Returns the scores from the database.
    :return: Dictionary with the relevant scores.
    """
    return {
        'training_score': database['training_score'],
        'real_training_score': database['training_data']['real_training_score'],
        'real_test_score': database['real_test_score']
    }


def get_table(kind):
    """
    This method will return the table (np array) of training data including the headers (names of the features).
    Used for displaying the data.
    """
    X_train, X_test, y_train, y_test, headers = get_data(kind)

    full_headers = ['Index']
    full_headers.extend(headers)

    index = np.arange(len(X_train), dtype=int).reshape(len(X_train), 1)
    # convert from list to vertical array
    y_tr = np.array(y_train).reshape(len(y_train), 1)

    table = np.hstack((np.hstack((index, y_tr)), X_train))

    return [table, full_headers]


def numericalize(df):
    """
    Convert all the columns with categorical data to numeric data by exchanging the alphanumeric categories with their
    numeric codes.
    :param df: pandas dataframe
    :return: numpy array
    """
    global database

    cat_columns = df.select_dtypes(['category']).columns

    # save a dict with the codes and equivalent categories for each cat feature
    mapping = {}
    for cat_column in cat_columns:
        mapping[cat_column] = dict(enumerate(df[cat_column].cat.categories))

    # save mapping to database
    database['mapping'] = mapping

    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    return df


def impute(df, y, strategy, headers=None):
    """
    Perform operations on the data such as dropping rows/columns with missing values or imputation of NaNs.
    """
    if strategy not in ['drop_row', 'drop_col', 'mean', 'median']:
        raise Exception(f'Given param strategy -> {strategy} <- not known.')

    if strategy == 'drop_row':
        # also need to drop some labels in y, thus stack X and y
        # first convert y from list to vertical np array
        # y = np.array(y).reshape(len(y), 1)
        df['label'] = y
        df.dropna(axis=0, inplace=True)
        y = df['label']
        df = df.drop('label', axis=1)

        return df, y, headers

    if strategy == 'drop_col':
        if headers is not None:
            seen_cols = []
            to_pop = []
            # remove column with NaNs from headers list
            for (row, col), value in np.ndenumerate(df):
                # to not drop ints. pandas gotcha
                if np.isnan(value) and col not in seen_cols and not isinstance(df.iloc[row, col], int):
                    to_pop.append(col + 1)
                    seen_cols.append(col)
            # sort cols (larger numbers at the end) and reverse list to pop those cols first
            to_pop.sort()
            to_pop.reverse()
            for header in to_pop:
                headers.pop(header)

        df.dropna(axis=1, inplace=True)

        return df, y, headers

    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df = imp.fit_transform(df)

    return df, y, headers


def build_tree(X_train, y_train, max_features=None, min_samples_leaf=1, max_depth=None):
    """
    Build a DecisionTreeRegressor and return the tree plus its score on the training set.
    """
    if X_train.dtype not in [np.float64, np.int32]:
        raise Exception('Input array X_train is not all numerical')

    m = DecisionTreeRegressor(max_features=max_features, min_samples_leaf=min_samples_leaf, max_depth=max_depth)

    m.fit(X_train, y_train)
    score = m.score(X_train, y_train)

    return m, score


def preprocess(strategy):
    """
    Will simulate the entire preprocessing for one chosen strategy.
    :param strategy: must either be 'raw_data' or 'processed_data'
    """
    df_train, df_test, y_train, y_test, headers = get_data('raw_data', unpack=False)

    # convert categorical data to their numeric codes for further processing
    df_train = numericalize(df_train)
    df_test = numericalize(df_test)

    # !!! only process headers once (otherwise certain indexes in list will be dropped again) !!!
    df_train, y_train, headers = impute(df_train, y_train, strategy, headers)

    df_test, y_test, _ = impute(df_test, y_test, strategy)

    # something returns an array already
    if not isinstance(df_train, (np.ndarray, np.generic) ):
        X_train, X_test = df_train.values, df_test.values
    # if is np array
    else:
        X_train, X_test = df_train, df_test
        df_train, df_test = pd.DataFrame(df_train), pd.DataFrame(df_test)

    # try the preprocessed training data with a tree
    m, score = build_tree(X_train, y_train)

    # save the data to the database as preprocessed data
    for set, path in [(df_train, 'df_train'), (df_test, 'df_test'), (y_train, 'y_train'), (y_test, 'y_test'),
                      (headers, 'headers')]:
        database['preprocessed_data'][path] = set

    database['training_score'] = score

    return


def training(max_features, min_samples_leaf, max_depth):
    """
    The whole training procedure is done in this method.
    """
    global database
    filename = get_hash()
    database['dtree_hash'] = filename

    X_train, X_test, y_train, y_test, headers = get_data('preprocessed_data')

    # remove label from headers
    headers.pop(0)

    m, real_training_score = build_tree(X_train, y_train, max_features=max_features,
                                        min_samples_leaf=min_samples_leaf, max_depth=max_depth)

    real_test_score = m.score(X_test, y_test)

    graph = Source(export_graphviz(m, out_file=None, feature_names=headers, filled=True, rounded=True,
                                   special_characters=True, precision=3))
    png_bytes = graph.pipe(format='png')
    with open(f'static/pictures/{filename}.png', 'wb') as file:
        file.write(png_bytes)

    # save the model
    database['training_data']['DecisionTreeRegressor'] = m
    database['training_data']['real_training_score'] = real_training_score
    database['training_data']['max_features'] = max_features
    database['training_data']['min_samples_leaf'] = min_samples_leaf
    database['training_data']['max_depth'] = max_depth

    database['real_test_score'] = real_test_score

    return real_training_score, real_test_score


def get_train_filename():
    """
    Return the full filename for the latest .png picture of a dtree
    """

    return f"/static/pictures/{database['dtree_hash']}.png"


def get_data_structure():
    """
    Find out about the maximum and minimum values for each feature.
    """

    [X_train, X_test, y_train, y_test, headers] = get_data('preprocessed_data')

    # slice off the label
    feature_names = headers[1:]

    X = np.vstack((X_train, X_test))

    mins = list(np.nanmin(X, axis=0))
    maxs = list(np.nanmax(X, axis=0))

    steps = [abs((maxs[i] - mins[i]) / 100) for i in range(len(mins))]

    if len(mins) != len(maxs) or len(maxs) != len(feature_names):
        raise Exception(f'Length of mins, {len(mins)} is not equal to length of maxs, '
                        f'{len(maxs)} or length of feature names, {len(feature_names)}')

    return [[feature_names[i], mins[i], maxs[i], steps[i]] for i in range(len(feature_names))]


def get_mapping():
    return database['mapping']


def get_slider_config():
    """
    Create a dictionary which can be looped through in the html template to produce a dynamic number of sliders
    for the deployment section.
    """
    a = get_data_structure()

    config = []

    for [header, min_, max_, step] in a:
        is_cat = False
        codes = None
        if isinstance(feature_layout[header], Categorical):
            codes = get_mapping()[header]
            is_cat = True
            step = 1
        slider = {
            'name': header,
            'is_cat': is_cat,
            'codes': codes,
            'slider_id': f'{header}_slider',
            'value_id': f'{header}_value',
            'min': min_,
            'max': max_,
            'step': step
        }
        config.append(slider)

    return config


def make_prediction(sample):
    """
    Given a single sample as data input return the prediction of the DecisionTreeRegressor.
    """
    real_training_score, regressor, max_features, min_samples_leaf, max_depth = get_data('training_data')

    # reshape because of single sample
    x = np.array(sample).reshape(1, -1)

    pred = regressor.predict(x)[0]

    # save the sample and prediction into database
    database['prediction'] = pred
    database['sample'] = sample

    return pred


def get_sample_pred():
    """
    For displaying sample and prediction in the deployment section.
    """
    feature_names = database['preprocessed_data']['headers'][1:]
    sample = database['sample']
    if sample is not None:
        sample = [(feature_names[i], sample[i]) for i in range(len(feature_names))]
    return sample, database['prediction']


def decision_tree_path(sample, regressor, headers):
    """
    Save a .png image of the decision tree with a highlighted path for the single trained sample.
    """

    # make sample array 2D
    sample = [sample]

    dot_data = export_graphviz(regressor, out_file=None,
                               feature_names=headers,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    # empty all nodes, i.e.set color to white and number of samples to zero
    # basically change encoding to design preference (white does not fit here)
    for node in graph.get_node_list():
        if node.get_attributes().get('label') is None:
            continue
        if 'samples = ' in node.get_attributes()['label']:
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = 0'
            node.set('label', '<br/>'.join(labels))
            # node.set_fillcolor('white')

    decision_paths = regressor.decision_path(sample)

    for decision_path in decision_paths:
        for n, node_value in enumerate(decision_path.toarray()[0]):
            if node_value == 0:
                continue
            node = graph.get_node(str(n))[0]
            node.set_fillcolor('green')
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

            node.set('label', '<br/>'.join(labels))

    return graph


def deploy(sample):
    """
    The deployment process.
    """
    global database
    filename = get_hash()
    database['dtree_path_hash'] = filename

    # find out, if features have been dropped
    if len(database['raw_data']['headers']) != len(database['preprocessed_data']['headers']):
        headers = get_data('preprocessed_data')[4][1:]
    else:
        headers = get_data('raw_data')[4][1:]

    regressor = get_data('training_data')[1]
    make_prediction(sample)

    graph = decision_tree_path(sample, regressor, headers)
    full_filename = f'static/pictures/{filename}.png'
    graph.write_png(full_filename)

    return


def get_deploy_filename():
    """
    Return the full filename for the latest .png picture of a path of a sample through a dtree.
    """

    return f"/static/pictures/{database['dtree_path_hash']}.png"


def get_hash():
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(12))

