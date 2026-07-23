from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import FloatField, SubmitField
from wtforms.validators import NumberRange

class UploadForm(FlaskForm):

    content = FileField(
        "Content Image",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "jpeg", "png"], "Images only!")
        ]
    )

    style = FileField(
        "Style Image",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "jpeg", "png"], "Images only!")
        ]
    )

    alpha = FloatField(
        "Style Strength",
        default=1.0,
        validators=[NumberRange(min=0, max=1)]
    )

    submit = SubmitField("Generate Style")