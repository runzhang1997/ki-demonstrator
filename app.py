from flask import Flask, render_template, flash, url_for, redirect, request, \
    jsonify
from utils.generator import DataGenerator
import os
import json
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
from utils.tree_export import tree_to_json


def rules(clf, features, labels, node_index=0):
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        # count_labels = zip(clf.tree_.value[node_index, 0], labels)
        # node['name'] = ', '.join(('{} of {}'.format(int(count), label)
        #                          for count, label in count_labels))
        node['type'] = 'leaf'
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        node['error'] = np.float64(clf.tree_.impurity[node_index]).item()
        node['samples'] = clf.tree_.n_node_samples[node_index]
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['type'] = 'split'
        node['label'] = '{} > {}'.format(feature, threshold)
        node['error'] = np.float64(clf.tree_.impurity[node_index]).item()
        node['samples'] = clf.tree_.n_node_samples[node_index]
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]

    return node


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


PICTURE_FOLDER = os.path.join('static', 'pictures')

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['UPLOAD_FOLDER'] = PICTURE_FOLDER

data_generator = DataGenerator()

@app.route('/')
@app.route('/intro/')
def introduction():
    return render_template('intro.html', current_page='introduction')


@app.route('/dummy_acquire_data/', methods=['GET', 'POST'])
def dummy_acquire_data():
    df_X, df_y = data_generator.get_data(0)

    headers = np.hstack((df_X.columns, df_y.columns))

    table = np.hstack((df_X.values, df_y.values))

    n_samples, n_features = df_X.shape

    return render_template('dummy_acquire_data.html', current_page='aquisition',
                           table=table,
                           headers=headers, n_samples=n_samples,
                           n_features=n_features, progress=25,
                           responsibility=["Domänenexperte"])


@app.route('/dummy_preprocessing/', methods=['GET', 'POST'])
def dummy_preprocessing():
    preprocessing_step = 0

    if request.args.get("step") != None:
        preprocessing_step = int(request.args.get("step"))

    df_X, df_y = data_generator.get_data(preprocessing_step)

    headers = np.hstack((df_X.columns, df_y.columns))

    table = np.hstack((df_X.values, df_y.values))

    n_samples, n_features = df_X.shape

    if preprocessing_step == 0:
        return render_template('dummy_preprocessing.html',
                               current_page='preprocessing', table=table,
                               headers=headers, n_samples=n_samples,
                               n_features=n_features, progress=40,
                               responsibility=["Domänenexperte", "KI-Experte"])
    elif preprocessing_step == 1:
        return render_template('dummy_preprocessing_nan_hidden.html',
                               current_page='preprocessing', table=table,
                               headers=headers, n_samples=n_samples,
                               n_features=n_features, progress=60,
                               responsibility=["Domänenexperte", "KI-Experte"])
    elif preprocessing_step == 2:
        return render_template('dummy_preprocessing_one_hot.html',
                               current_page='preprocessing', table=table,
                               headers=headers, n_samples=n_samples,
                               n_features=n_features, progress=75,
                               responsibility=["Domänenexperte", "KI-Experte"])


@app.route('/dummy_training/', methods=['GET', 'POST'])
def dummy_training():
    if all(k in request.form for k in
           ['max_depth', 'min_samples_leaf', 'max_features']):

        max_depth = int(request.form['max_depth'])
        max_features = float(request.form['max_features'])
        min_samples_leaf = int(request.form['min_samples_leaf'])

    else:
        max_depth = None
        max_features = None
        min_samples_leaf = 1

    regressor = DecisionTreeRegressor(max_features=max_features,
                                      min_samples_leaf=min_samples_leaf,
                                      max_depth=max_depth)

    df_X, df_y = data_generator.get_data(2)

    regressor = regressor.fit(df_X, df_y)

    json_data = tree_to_json(regressor)

    # json_data = {"error": 42716.2954, "samples": 506, "value": [22.532806324110698],
    #        "label": "RM <= 6.94", "type": "split", "children": [
    #        {"error": 17317.3210, "samples": 430, "value": [19.93372093023257],
    #         "label": "LSTAT <= 14.40", "type": "leaf"},
    #        {"error": 6059.4193, "samples": 76, "value": [37.23815789473684],
    #         "label": "RM <= 7.44", "type": "leaf"}]}
    try:
        with open('static/output.json', 'w') as outfile:
            json.dump(json_data, outfile)
    except IOError:
        print("succ")

    n_samples, n_features = df_X.shape

    return render_template('dummy_training.html', current_page='training',
                           tree_data=json_data, n_samples=n_samples,
                           n_features=n_features, progress=90,
                           responsibility=["KI-Experte"])


@app.route('/dummy_deployment/', methods=['GET', 'POST'])
def dummy_deployment():
    df_X, df_y = data_generator.get_data(2)
    _, n_features = df_X.shape
    json_data = {"error": 42716.2954, "samples": 506,
                 "value": [22.532806324110698],
                 "label": "RM <= 6.94", "type": "split", "children": [
            {"error": 17317.3210, "samples": 430, "value": [19.93372093023257],
             "label": "LSTAT <= 14.40", "type": "leaf"},
            {"error": 6059.4193, "samples": 76, "value": [37.23815789473684],
             "label": "RM <= 7.44", "type": "leaf"}]}

    return render_template('dummy_deployment.html', current_page='deployment',
                           tree_data=json_data, n_samples=None, n_features=None,
                           progress=100, responsibility=["Domänenexperte"])


if __name__ == '__main__':
    app.run(debug=True)
