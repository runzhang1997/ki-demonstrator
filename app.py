from flask import Flask, render_template, flash, url_for, redirect, request, \
    jsonify
from utils.generator import DataGenerator
import os
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
from utils.tree_export import tree_to_json


PICTURE_FOLDER = os.path.join('static', 'pictures')

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['UPLOAD_FOLDER'] = PICTURE_FOLDER

data_generator = DataGenerator()


@app.route('/')
@app.route('/intro/')
def introduction():
    return render_template('intro.html', title='Intro')


@app.route('/acquire_data/', methods=['GET', 'POST'])
def acquire_data():
    df_X, df_y = data_generator.get_data(0)

    headers = np.hstack((df_X.columns, df_y.columns))

    table = np.hstack((df_X.values, df_y.values))

    n_samples = df_X.shape[0]

    return render_template('acquire_data.html', table=table,
                           headers=headers, n_samples=n_samples)


@app.route('/preprocessing/', methods=['GET', 'POST'])
def preprocessing():

    preprocessing_step = 0

    if preprocessing_step in request.form:
        preprocessing_step = int(request.form["preprocessing_step"])

    df_X, df_y = data_generator.get_data(preprocessing_step)

    headers = np.hstack((df_X.columns, df_y.columns))

    table = np.hstack((df_X.values, df_y.values))

    n_samples = df_X.shape[0]

    return render_template('preprocessing.html', table=table,
                           headers=headers, n_samples=n_samples)


@app.route('/training/', methods=['GET', 'POST'])
def training():

    if all(k in request.form for k in ['max_depth', 'min_samples_leaf', 'max_features']):

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

    # regressor = regressor.fit(df_X, df_y)

    # json_data = tree_to_json(regressor)

    json_data = {"error": 42716.2954, "samples": 506, "value": [22.532806324110698],
            "label": "RM <= 6.94", "type": "split", "children": [
            {"error": 17317.3210, "samples": 430, "value": [19.93372093023257],
             "label": "LSTAT <= 14.40", "type": "leaf"},
            {"error": 6059.4193, "samples": 76, "value": [37.23815789473684],
             "label": "RM <= 7.44", "type": "leaf"}]}

    n_samples = df_X.shape[0]

    return render_template('training.html', tree_data=json_data, n_samples=n_samples)


@app.route('/deployment/', methods=['GET', 'POST'])
def deployment():

    return render_template('deployment.html')


if __name__ == '__main__':
    app.run(debug=True)
