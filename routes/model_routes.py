import pandas as pd

from flask import Blueprint
from flask import render_template


model = Blueprint(
    "model",
    __name__
)


@model.route("/models")
def models():

    df = pd.read_csv(
        "reports/model_comparison.csv"
    )

    table = df.to_dict(
        orient="records"
    )

    best_model = df.loc[
        df["Accuracy"].idxmax()
    ]

    return render_template(
        "models.html",
        table=table,
        best_model=best_model
    )
