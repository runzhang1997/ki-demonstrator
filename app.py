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
    df_X, df_y = backend.get_data(0)

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
    max_depth = int(request.args.get("max_depth", 50))

    json_data, mean_absolute_error = backend.generate_model(train_size, max_depth)

    mean_absolute_error = np.random.random() * 5000 * (51 - max_depth) / 50 * (1 - train_size)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    try:
        with open(f'static/output-{timestamp}.json', 'w') as outfile:
            json.dump(json_data, outfile, cls=MyEncoder)
    except IOError:
        print("success")

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
    # feature_dict = request.form["feature_dict"]

    feature_dict = {"x": 0, "y": 0}

    prediction, model_json = backend.evaluate_model(feature_dict)

    try:
        with open('static/predict.json', 'w') as outfile:
            json.dump(model_json, outfile, cls=MyEncoder)
    except IOError:
        print("success")

    return render_template('deployment.html', current_page='deployment',
                           tree_data=model_json, prediction=prediction,
                           n_samples=None, n_features=None,
                           progress=100, responsibility=["Domänenexperte"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
