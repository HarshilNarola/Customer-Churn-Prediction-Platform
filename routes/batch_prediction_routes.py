import os

from flask import Blueprint
from flask import render_template
from flask import request
from flask import send_file

from controllers.batch_prediction_controller import (
    BatchPredictionController
)


batch_prediction = Blueprint(
    "batch_prediction",
    __name__
)

controller = BatchPredictionController()


@batch_prediction.route(
    "/batch-predict",
    methods=["GET", "POST"]
)
def batch_predict():

    # ==========================================================
    # GET Request
    # ==========================================================

    if request.method == "GET":

        return render_template(
            "batch_predict.html"
        )

    # ==========================================================
    # Check File Upload
    # ==========================================================

    if "file" not in request.files:

        return render_template(

            "error.html",

            message="No file was uploaded."

        )

    file = request.files["file"]

    if file.filename == "":

        return render_template(

            "error.html",

            message="Please select a CSV file."

        )

    # ==========================================================
    # Validate File Type
    # ==========================================================

    if not file.filename.lower().endswith(".csv"):

        return render_template(

            "error.html",

            message="Only CSV files are allowed."

        )

    # ==========================================================
    # Save Uploaded File
    # ==========================================================

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    os.makedirs(
        "downloads",
        exist_ok=True
    )

    filepath = os.path.join(

        "uploads",

        file.filename

    )

    file.save(filepath)

    # ==========================================================
    # Process CSV
    # ==========================================================

    output_file, missing_columns = controller.process(
        filepath
    )

    if missing_columns:

        return render_template(

            "error.html",

            message="Missing Columns: " +
            ", ".join(missing_columns)

        )

    # ==========================================================
    # Download Prediction File
    # ==========================================================

    return send_file(

        output_file,

        as_attachment=True,

        download_name="prediction_results.csv"

    )