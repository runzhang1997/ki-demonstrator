from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, FloatField, SelectField, ValidationError
from wtforms.validators import DataRequired, NumberRange, InputRequired

import utils


class LessOrEqualThan(object):
    """
    WTForms custom validator. Compares the values of two fields.
    """
    def __init__(self, fieldname, message=None):
        self.fieldname = fieldname
        self.message = message

    def __call__(self, form, field):
        try:
            other = form[self.fieldname]
        except KeyError:
            raise ValidationError(field.gettext("Invalid field name '%s'.") % self.fieldname)
        if field.data > other.data:
            d = {
                'other_label': hasattr(other, 'label') and other.label.text or self.fieldname,
                'other_name': self.fieldname
            }
            message = self.message
            if message is None:
                message = field.gettext('Field must be less or equal to %(other_label)s.')

            raise ValidationError(message % d)


class RequestDataForm(FlaskForm):

    feature_count = IntegerField('Number of features', validators=[DataRequired(),
                                                                   NumberRange(min=1, max=utils.get_max_features())])

    effective_rank = IntegerField('Effective Rank', validators=[DataRequired(), LessOrEqualThan('feature_count')])

    noise = FloatField('Noise', validators=[DataRequired(), NumberRange(min=0, max=100)])

    submit = SubmitField('Submit')


class PreprocessingForm(FlaskForm):

    strategy = SelectField('Strategy', choices=[('drop_row', 'Drop rows with NaN'),
                                                ('drop_col', 'Drop columns with NaN'), ('mean', 'Impute with mean'),
                                                ('median', 'Impute with median')], validators=[InputRequired()])

    submit = SubmitField('Submit')
