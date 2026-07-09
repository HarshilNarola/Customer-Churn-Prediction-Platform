from pathlib import Path

from flask import (
    Blueprint,
    render_template,
    request,
    send_file
)

from controllers.batch_prediction_controller import (
    BatchPredictionController
)


batch_prediction = Blueprint(
    "batch_prediction",
    __name__
)

controller = BatchPredictionController()

UPLOAD_DIRECTORY = Path("uploads")
DOWNLOAD_DIRECTORY = Path("downloads")

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

DOWNLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


@batch_prediction.route(
    "/batch-predict",
    methods=["GET", "POST"]
)
def batch_predict():
    """
    Handle batch customer churn prediction using
    an uploaded CSV file.
    """

    # ==========================================================
    # GET Request
    # ==========================================================

    if request.method == "GET":

        return render_template(
            "batch_predict.html"
        )

    # ==========================================================
    # Validate Uploaded File
    # ==========================================================

    uploaded_file = request.files.get("file")

    if uploaded_file is None:

        return render_template(

            "error.html",

            message="No file was uploaded."

        )

    if uploaded_file.filename == "":

        return render_template(

            "error.html",

            message="Please select a CSV file."

        )

    if not uploaded_file.filename.lower().endswith(".csv"):

        return render_template(

            "error.html",

            message="Only CSV files are supported."

        )

    # ==========================================================
    # Save File
    # ==========================================================

    file_path = UPLOAD_DIRECTORY / uploaded_file.filename

    uploaded_file.save(
        file_path
    )

    # ==========================================================
    # Process Prediction
    # ==========================================================

    try:

        output_file, missing_columns = controller.process(
            str(file_path)
        )

    except Exception as error:

        return render_template(

            "error.html",

            message=str(error)

        )

    if missing_columns:

        return render_template(

            "error.html",

            message=(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        )

    # ==========================================================
    # Download Prediction File
    # ==========================================================

    return send_file(

        output_file,

        as_attachment=True,

        download_name="prediction_results.csv"

    )