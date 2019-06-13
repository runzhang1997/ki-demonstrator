from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import numpy as np

from graphviz import Source

import os
import glob


# initialize the database with None values
database = {
    'raw_data': {
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'headers': None
    },
    'preprocessed_data': {
        'X_train': None,
        'X_test': None,
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
    'real_test_score': None
}


# this id is for naming .png files of dtrees for individual training procedures
train_id = 0


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
    directory = '/static/pictures/'
    os.chdir(directory)
    files = glob.glob('*.png')
    for filename in files:
        os.unlink(filename)

    # and reset the train_id to 0
    train_id = 0

    return


def get_data(kind):
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

        # if regressor is not None:
        #     real_training_score, regressor, max_features, min_samples_leaf, max_depth = real_training_score.copy(), regressor.copy(), max_features.copy(), min_samples_leaf.copy(), max_depth.copy()

        return [real_training_score, regressor, max_features, min_samples_leaf, max_depth]

    if kind not in ['raw_data', 'preprocessed_data']:
        raise Exception('This param kind is not known. (Does not point to any data)')

    data = database[kind]
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    headers = data['headers']

    # Return copies of the variables in the database!
    if X_train is not None:
        X_train, X_test, y_train, y_test, headers = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), headers.copy()

    return [X_train, X_test, y_train, y_test, headers]


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


def gen_data(n_train, n_test, n_features, effective_rank, noise):
    """
    This method creates the regression dataset with (effective rank) number of singular vectors.
    Some values in some columns will be set to NaN.
    """

    # first reset the database and picture subfolder
    reset()

    X, y, coef = make_regression(n_samples=n_train+n_test, n_features=n_features, effective_rank=effective_rank, noise=noise, coef=True)

    # decide on the columns which will hold NaNs
    col_len = X.shape[1]
    to_nan = []
    for i in range(int(col_len/3)):
        to_nan.append(np.random.randint(0, col_len))

    # set NaNs
    for (row, col), value in np.ndenumerate(X):
        if col in to_nan and np.random.random() < 0.1:
            X[row, col] = np.nan

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=42)

    # create headers for the data
    headers = ['Label']
    for i in range(X.shape[1]):
        headers.append(f'Feature {i}')

    # save the data so the database as raw data
    for set, path in [(X_train, 'X_train'), (X_test, 'X_test'), (y_train, 'y_train'), (y_test, 'y_test'), (headers, 'headers')]:
        database['raw_data'][path] = set

    return


def set_data(n_train, n_test, n_features, effective_rank, noise):
    """
    This method will save the generated dataset into global variables
    """
    global X_train, X_test, y_train, y_test, headers

    X_train, X_test, y_train, y_test, headers = gen_data(n_train, n_test, n_features, effective_rank, noise)

    return


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


def impute(X, y, strategy, headers=None):
    """
    Perform operations on the data such as dropping rows/columns with missing values or imputation of NaNs.
    """
    if strategy not in ['drop_row', 'drop_col', 'mean', 'median']:
        raise Exception(f'Given param strategy -> {strategy} <- not known.')

    if strategy == 'drop_row':
        # also need to drop some labels in y, thus stack X and y
        # first convert y from list to vertical np array
        y = np.array(y).reshape(len(y), 1)
        X = np.hstack((y, X))
        X = X[~np.isnan(X).any(axis=1)]
        # seperate X and y
        y = X[:, :1]
        X = X[:, 1:]

        return X, y, headers

    if strategy == 'drop_col':
        if headers is not None:
            seen_cols = []
            to_pop = []
            # remove column with NaNs from headers list
            for (row, col), value in np.ndenumerate(X):
                if np.isnan(value) and col not in seen_cols:
                    to_pop.append(col + 1)
                    # headers.pop(col + 1)
                    seen_cols.append(col)
            # sort cols (larger numbers at the end) and reverse list to pop those cols first
            to_pop.sort()
            to_pop.reverse()
            for i in to_pop:
                headers.pop(i)

        X = X[:, ~np.any(np.isnan(X), axis=0)]

        return X, y, headers

    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    X = imp.fit_transform(X)

    return X, y, headers


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
    X_train, X_test, y_train, y_test, headers = get_data('raw_data')

    # !!! only process headers once (otherwise certain indexes in list will be dropped again) !!!
    X_train, y_train, headers = impute(X_train, y_train, strategy, headers)

    X_test, y_test, _ = impute(X_test, y_test, strategy)

    # try the preprocessed training data with a tree
    m, score = build_tree(X_train, y_train)

    # save the data to the database as preprocessed data
    for set, path in [(X_train, 'X_train'), (X_test, 'X_test'), (y_train, 'y_train'), (y_test, 'y_test'),
                      (headers, 'headers')]:
        database['preprocessed_data'][path] = set

    database['training_score'] = score

    return


def training(max_features, min_samples_leaf, max_depth):
    """
    The whole training procedure is done in this method.
    """

    global train_id
    train_id += 1

    X_train, X_test, y_train, y_test, headers = get_data('preprocessed_data')

    # remove label from headers
    headers.pop(0)

    m, real_training_score = build_tree(X_train, y_train, max_features=max_features, min_samples_leaf=min_samples_leaf, max_depth=max_depth)

    graph = Source(export_graphviz(m, out_file=None, feature_names=headers, filled=True, special_characters=True, rotate=True, precision=3))
    png_bytes = graph.pipe(format='png')
    with open(f'static/pictures/dtree{train_id}.png', 'wb') as file:
        file.write(png_bytes)

    # save the model
    database['training_data']['DecisionTreeRegressor'] = m
    database['training_data']['real_training_score'] = real_training_score
    database['training_data']['max_features'] = max_features
    database['training_data']['min_samples_leaf'] = min_samples_leaf
    database['training_data']['max_depth'] = max_depth

    return real_training_score


def get_filename():
    """
    Return the full filename for the latest .png picture of a dtree
    """

    return f'/static/pictures/dtree{train_id}.png'

