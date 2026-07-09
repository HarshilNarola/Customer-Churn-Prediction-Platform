from pathlib import Path

import pandas as pd

from flask import (
    Blueprint,
    render_template
)


model = Blueprint(
    "model",
    __name__
)

MODEL_COMPARISON_REPORT = Path(
    "reports/model_comparison.csv"
)

FINAL_MODEL_REPORT = Path(
    "reports/final_model_report.csv"
)


@model.route("/models")
def models():
    """
    Display model comparison results and
    information about the final deployed model.
    """

    # ==========================================================
    # Validate Report Files
    # ==========================================================

    if not MODEL_COMPARISON_REPORT.exists():

        return render_template(

            "error.html",

            message="Model comparison report was not found."

        )

    if not FINAL_MODEL_REPORT.exists():

        return render_template(

            "error.html",

            message="Final model report was not found."

        )

    # ==========================================================
    # Load Model Comparison
    # ==========================================================

    comparison_dataframe = pd.read_csv(

        MODEL_COMPARISON_REPORT

    )

    comparison_table = comparison_dataframe.to_dict(

        orient="records"

    )

    # ==========================================================
    # Load Final Model Information
    # ==========================================================

    final_model_dataframe = pd.read_csv(

        FINAL_MODEL_REPORT

    )

    best_model_information = (

        final_model_dataframe.iloc[0].to_dict()

    )

    # ==========================================================
    # Render Page
    # ==========================================================

    return render_template(

        "models.html",

        table=comparison_table,

        best_model_info=best_model_information

    )