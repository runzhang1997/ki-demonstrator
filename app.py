from flask import Flask, render_template, flash, url_for, redirect, request, jsonify
import utils
import os
import numpy as np

PICTURE_FOLDER = os.path.join('static', 'pictures')

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['UPLOAD_FOLDER'] = PICTURE_FOLDER

data_generator = utils.DataGenerator()

@app.route('/')
@app.route('/intro/')
def introduction():
    return render_template('intro.html', title='Intro')


@app.route('/acquire_data/', methods=['GET', 'POST'])
def acquire_data():
    df_X, df_y = data_generator.get_raw_data()

    # print(df_X.shape, df_y.shape)

    headers = df_X.columns + df_y.columns

    table = np.hstack((df_X.values, df_y.values))

    # print(table.shape)

    n_samples, n_features = df_X.shape

    return render_template('acquire_data.html', table=table,
                           headers=headers, n_samples=n_samples,
                           n_features=n_features)


@app.route('/preprocessing/', methods=['GET', 'POST'])
def preprocessing():
    df_X, df_y = data_generator.get_raw_data()

    headers = df_X.columns + df_y.columns

    table = np.hstack((df_X.values, df_y.values))

    # print(table.shape)

    strategy = data_generator.preprocessing_stategy

    means = df_X.mean()

    n_samples, n_features = df_X.shape

    return render_template('preprocessing.html', table=table,
                           headers=headers, n_samples=n_samples,
                           n_features=n_features, strategy=strategy, means=means)


@app.route('/training/', methods=['GET', 'POST'])
def training():
    """WTForms does not supply a good form with sliders. Therefore a different structure with html and JavaScript
    sliders is being used for the inputs here."""

    if request.method == 'POST':
        flash('Successfully trained DecisionTreeRegressor.', 'success')
        max_depth = int(request.form['max_depth'])
        min_samples_leaf = int(request.form['min_samples_leaf'])
        max_features = float(request.form['max_features'])

        utils.training(max_features, min_samples_leaf, max_depth)

        return redirect(url_for('training'))

    # if data has not been generated before
    if any(i is None for i in utils.get_data('raw_data')):
        flash('Please get dataset before continuing.', 'danger')
        return redirect(url_for('acquire_data'))

    # if data has not been preprocessed before
    elif any(i is None for i in utils.get_data('preprocessed_data')):
        flash('Cannot train DecisionTreeRegressor without preprocessing.', 'danger')
        return redirect(url_for('preprocessing'))

    # if the model has been trained before (check if the real_training_score is None)
    elif utils.get_data('training_data')[0] is not None:
        real_training_score = utils.get_scores()['real_training_score']
        real_test_score = utils.get_scores()['real_test_score']
        n_samples = len(utils.get_table('preprocessed_data')[0])

        max_features, min_samples_leaf, max_depth = utils.get_data('training_data')[2:]

        full_filename = utils.get_train_filename()

        return render_template('training.html', title='Training', full_filename=full_filename,
                               training_score=real_training_score, test_score=real_test_score, n_samples=n_samples,
                               max_features=max_features, min_samples_leaf=min_samples_leaf, max_depth=max_depth)

    # if the model is to be trained next
    else:
        n_samples = len(utils.get_table('preprocessed_data')[0])
        return render_template('training.html', title='Training', n_samples=n_samples)


@app.route('/deployment/', methods=['GET', 'POST'])
def deployment():

    if request.method == 'POST':
        flash('Predicted sample with given feature values.', 'success')
        sliders = utils.get_slider_config()
        sample = []
        for slider in sliders:
            slider_name = slider['name']
            feature_val = float(request.form[slider_name])
            sample.append(feature_val)

        utils.deploy(sample)

        return redirect(url_for('deployment'))

    # if data has not been generated before
    if any(i is None for i in utils.get_data('raw_data')):
        flash('Please get dataset before continuing.', 'danger')
        return redirect(url_for('acquire_data'))

    # if data has not been preprocessed before
    elif any(i is None for i in utils.get_data('preprocessed_data')):
        flash('Cannot train DecisionTreeRegressor without preprocessing.', 'danger')
        return redirect(url_for('preprocessing'))

    # if the model has not been trained before
    elif utils.get_data('training_data')[0] is None:
        flash('You need to train the DecisionTreeRegressor first.', 'danger')
        return redirect(url_for('training'))

    else:
        sliders = utils.get_slider_config()
        # sample is a list of tuples
        sample, prediction = utils.get_sample_pred()
        filename = utils.get_deploy_filename()
        return render_template('deployment.html', title='Deployment', sliders=sliders,
                               sample=sample, prediction=prediction, filename=filename)


if __name__ == '__main__':
    app.run(debug=True)