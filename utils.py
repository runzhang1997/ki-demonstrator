from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import numpy as np

from graphviz import Source
import pydotplus

import os
import string
import random


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
    'real_test_score': None,
    # the last sample and predicted value
    'sample': None,
    'prediction': None,
    # the last set hashes for the .png in training and deploying
    'dtree_hash': None,
    'dtree_path_hash': None
}


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

    # and reset the ids to 0
    train_id = 0
    deploy_id = 0

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
        X_train, X_test, y_train, y_test, headers = X_train.copy(), X_test.copy(), \
                                                    y_train.copy(), y_test.copy(), headers.copy()

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

    X, y, coef = make_regression(n_samples=n_train+n_test, n_features=n_features,
                                 effective_rank=effective_rank, noise=noise, coef=True)

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
    for set, path in [(X_train, 'X_train'), (X_test, 'X_test'), (y_train, 'y_train'),
                      (y_test, 'y_test'), (headers, 'headers')]:
        database['raw_data'][path] = set

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
    # find out, if features have been dropped
    if len(database['raw_data']['headers']) != len(database['preprocessed_data']['headers']):
        [X_train, X_test, y_train, y_test, headers] = get_data('preprocessed_data')
    else:
        [X_train, X_test, y_train, y_test, headers] = get_data('raw_data')

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


def get_slider_config():
    """
    Create a dictionary which can be looped through in the html template to produce a dynamic number of sliders
    for the deployment section.
    """
    a = get_data_structure()

    config = []

    for [header, min_, max_, step] in a:
        slider = {
            'name': header,
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
    return database['sample'], database['prediction']


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

