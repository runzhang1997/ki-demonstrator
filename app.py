from flask import Flask, render_template, flash, url_for, redirect, request
import utils
from forms import RequestDataForm, PreprocessingForm
import os

PICTURE_FOLDER = os.path.join('static', 'pictures')

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['UPLOAD_FOLDER'] = PICTURE_FOLDER


@app.route('/')
@app.route('/intro/')
def introduction():
    return render_template('intro.html', title='Intro')


@app.route('/acquire_data/', methods=['GET', 'POST'])
def acquire_data():
    form = RequestDataForm()

    if form.validate_on_submit():
        flash('Successfully created dataset.', 'success')
        feature_count = form.feature_count.data
        effective_rank = form.effective_rank.data
        noise = form.noise.data
        # hardcode sample size in training and test set
        utils.gen_data(1000, 200, feature_count, effective_rank, noise)
        table, headers = utils.get_table('raw_data')
        n_samples = len(table)
        return render_template('acquire_data.html', title='Acquire Data', table=table, headers=headers, form=form, n_samples=n_samples)

    # if data has been generated before, show the data
    if not any(i is None for i in utils.get_data('raw_data')):
        table, headers = utils.get_table('raw_data')
        n_samples = len(table)
        return render_template('acquire_data.html', title='Acquire Data', table=table, headers=headers, form=form, n_samples=n_samples)
    # if data has not been generated before
    else:
        return render_template('acquire_data.html', title='Acquire Data', form=form)


@app.route('/preprocessing/', methods=['GET', 'POST'])
def preprocessing():
    form = PreprocessingForm()

    if form.validate_on_submit():
        flash('Applied preprocessing strategy to dataset.', 'success')
        strategy = form.strategy.data
        utils.preprocess(strategy)
        training_score = utils.get_scores()['training_score']
        # get processed data
        table, headers = utils.get_table('preprocessed_data')
        n_samples = len(table)
        return render_template('preprocessing.html', title='Preprocessing', table=table, headers=headers, form=form, training_score=training_score, n_samples=n_samples)

    # if data has not been generated before
    if any(i is None for i in utils.get_data('raw_data')):
        flash('Please get dataset before continuing.', 'danger')
        return redirect(url_for('acquire_data'))
    # if data has not been preprocessed before
    elif any(i is None for i in utils.get_data('preprocessed_data')):
        # show raw data
        table, headers = utils.get_table('raw_data')
        n_samples = len(table)
        return render_template('preprocessing.html', title='Preprocessing', table=table, headers=headers, form=form, n_samples=n_samples)
    # if data has been preprocessed before
    else:
        table, headers = utils.get_table('preprocessed_data')
        n_samples = len(table)
        training_score = utils.get_scores()['training_score']
        return render_template('preprocessing.html', title='Preprocessing', table=table, headers=headers, form=form, training_score=training_score, n_samples=n_samples)


@app.route('/training/', methods=['GET', 'POST'])
def training():
    """WTForms does not supply a good form with sliders. Therefore a different structure with html and JavaScript
    sliders is being used for the inputs here."""

    if request.method == 'POST':
        flash('Successfully trained DecisionTreeRegressor.', 'success')
        max_depth = int(request.form['max_depth'])
        min_samples_leaf = int(request.form['min_samples_leaf'])
        max_features = int(request.form['max_features'])

        real_training_score = utils.training(max_features, min_samples_leaf, max_depth)
        n_samples = len(utils.get_table('preprocessed_data')[0])

        # todo decide
        # full_filename = '/static/pictures/dtree.png'
        # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'dtree.png')
        full_filename = utils.get_filename()

        return render_template('training.html', title='Training', dtree_image=full_filename, training_score=real_training_score, n_samples=n_samples)

    # if data has not been generated before
    if any(i is None for i in utils.get_data('raw_data')):
        flash('Please get dataset before continuing.', 'danger')
        return redirect(url_for('acquire_data'))

    # if data has not been preprocessed before
    elif any(i is None for i in utils.get_data('preprocessed_data')):
        flash('Cannot train RandomForestRegressor without preprocessing.', 'danger')
        return redirect(url_for('preprocessing'))

    # if the model has been trained before (check if the real_training_score is None)
    elif utils.get_data('training_data')[0] is not None:
        real_training_score = utils.get_scores()['real_training_score']
        n_samples = len(utils.get_table('preprocessed_data')[0])

        # todo decide
        # full_filename = '/static/pictures/dtree.png'
        # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'dtree.png')
        full_filename = utils.get_filename()

        return render_template('training.html', title='Training', dtree_image=full_filename,
                               training_score=real_training_score, n_samples=n_samples)
    # if the model is to be trained next
    else:
        n_samples = len(utils.get_table('preprocessed_data')[0])
        return render_template('training.html', title='Training', n_samples=n_samples)


@app.route('/deployment/')
def deployment():
    return render_template('deployment.html', title='Deployment')


if __name__ == '__main__':
    app.run(debug=True)