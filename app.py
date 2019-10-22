#!/usr/bin/python
#-*- coding:utf-8 -*-

from flask import Flask, render_template, flash, url_for, redirect, request, \
    jsonify
from utils.backend import Backend
import os
import json
import numpy as np
import time

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

backend = Backend()


@app.route('/')
@app.route('/introduction/')
def introduction():
    return render_template('introduction.html', current_page='introduction')


@app.route('/acquisition/', methods=['GET', 'POST'])
def acquisition():
    data_acquisition_step = int(request.args.get("step", 0))
    print (data_acquisition_step)
    df_X, df_y = backend.get_data_acquisition(data_acquisition_step)

    if data_acquisition_step == 1 or data_acquisition_step == 2:
        headers = df_X.columns
        table = df_X.values
    else:
        headers = np.hstack((df_X.columns, df_y.columns))
        table = np.hstack((df_X.values, df_y.values))

    table = table[:100]

    n_samples, n_features = df_X.shape

    return render_template('acquisition.html', current_page='aquisition',
                           table=table,
                           headers=headers, n_samples=n_samples,
                           n_features=n_features, progress=25,
                           responsibility=["Domänenexperte"])


@app.route('/preprocessing/', methods=['GET', 'POST'])
def preprocessing():
    preprocessing_step = int(request.args.get("step", 0))

    df_X, df_y = backend.get_data(preprocessing_step)

    headers = np.hstack((df_X.columns, df_y.columns))

    table = np.hstack((df_X.values, df_y.values))

    table = table[:100]

    n_samples, n_features = df_X.shape

    if preprocessing_step == 0:
        return render_template('preprocessing.html',
                               current_page='preprocessing', table=table,
                               headers=headers, n_samples=n_samples,
                               n_features=n_features, progress=40,
                               responsibility=["Domänenexperte", "KI-Experte"])
    elif preprocessing_step == 1:
        return render_template('preprocessing_nan_hidden.html',
                               current_page='preprocessing', table=table,
                               headers=headers, n_samples=n_samples,
                               n_features=n_features, progress=60,
                               responsibility=["Domänenexperte", "KI-Experte"])
    elif preprocessing_step == 2:

        return render_template('preprocessing_one_hot.html',
                               current_page='preprocessing', table=table,
                               headers=headers, n_samples=n_samples,
                               n_features=n_features, progress=75,
                               responsibility=["Domänenexperte", "KI-Experte"])


@app.route('/training/', methods=['GET', 'POST'])
def training():
    train_size = float(request.args.get("train_size", 0.7))
    # min_samples_leaf = request.form.get("min_samples_leaf", 1)
    max_depth = int(request.args.get("max_depth", 100))

    json_data, mean_absolute_error = backend.generate_model(train_size, max_depth)



    # if train_size < .5:
    #     mean_absolute_error = np.random.random() * 2500 + 2500
    # elif max_depth < 20:
    #     mean_absolute_error = np.random.random() * 1500 + 500
    # else:
    #     mean_absolute_error = np.random.random() * 1000

    # mean_absolute_error = np.random.random() * 5000 * (51 - max_depth) / 50 * (1 - train_size)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # try:
    with open(f'static/output-{timestamp}.json', 'w') as outfile:
        json.dump(json_data, outfile, cls=MyEncoder)
    # except IOError:
    #     print("ERROR")

    while not os.path.exists(f'static/output-{timestamp}.json'):
        pass

    df_X, _ = backend.get_data(2)
    n_samples, n_features = df_X.shape

    return render_template('training.html', current_page='training', timestamp=timestamp,
                           tree_data=json_data, train_size=train_size, max_depth=max_depth,
                           mean_absolute_error=mean_absolute_error, n_samples=n_samples,
                           n_features=n_features, progress=90,
                           responsibility=["KI-Experte"])


@app.route('/deployment/', methods=['GET', 'POST'])
def deployment():
    if all(k in request.form for k in ['Anzahl der Kavitäten', 'Form der Kavitäten', 'Schieberanzahl', 'Kanaltyp']):
        feature_dict = {"Anzahl Kavitäten": float(request.form['Anzahl der Kavitäten']),
                        "Kavitätenform_A": (request.form['Form der Kavitäten']=='A'),
                        "Kavitätenform_B": (request.form['Form der Kavitäten']=='B'),
                        "Kavitätenform_C": (request.form['Form der Kavitäten']=='C'),
                        "Kavitätenform_D": (request.form['Form der Kavitäten']=='D'),
                        'Schieberanzahl': float(request.form['Schieberanzahl']),
                        'Kanaltyp_Heißkanal': (request.form['Kanaltyp']=='Heißkanal'),
                        'Kanaltyp_Kaltkanal': (request.form['Kanaltyp']=='Kaltkanal')
                        }
    else:
        feature_dict = {"Anzahl Kavitäten": 0,
                        "Kavitätenform_A": 0,
                        "Kavitätenform_B": 0,
                        "Kavitätenform_C": 0,
                        "Kavitätenform_D": 0,
                        'Schieberanzahl': 0,
                        'Kanaltyp_Heißkanal': 0,
                        'Kanaltyp_Kaltkanal': 0
                        }
    print (request.form)
    print (feature_dict)

    prediction, model_json = backend.evaluate_model(feature_dict)

    try:
        with open('static/predict.json', 'w') as outfile:
            json.dump(model_json, outfile, cls=MyEncoder)
    except IOError:
        print("ERROR")

    return render_template('deployment.html', current_page='deployment',
                           tree_data=model_json, prediction=prediction,
                           n_samples=None, n_features=None,
                           progress=100, responsibility=["Domänenexperte"])


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
    #app.run(host="0.0.0.0", debug=True)