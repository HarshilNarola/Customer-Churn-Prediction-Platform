from flask import Blueprint
from flask import render_template
from flask import request

from controllers.prediction_controller import PredictionController


prediction = Blueprint(
    "prediction",
    __name__
)

controller = PredictionController()


@prediction.route("/predict", methods=["GET", "POST"])
def predict():

    # ==========================================================
    # GET Request
    # ==========================================================

    if request.method == "GET":

        return render_template(
            "predict.html"
        )

    # ==========================================================
    # Read Form Data
    # ==========================================================

    form_data = {

        "Age": int(request.form["Age"]),
        "Gender": request.form["Gender"],
        "Tenure": int(request.form["Tenure"]),
        "Usage Frequency": int(request.form["Usage Frequency"]),
        "Support Calls": int(request.form["Support Calls"]),
        "Payment Delay": int(request.form["Payment Delay"]),
        "Subscription Type": request.form["Subscription Type"],
        "Contract Length": request.form["Contract Length"],
        "Total Spend": int(request.form["Total Spend"]),
        "Last Interaction": int(request.form["Last Interaction"])

    }

    # ==========================================================
    # Validation
    # ==========================================================

    if not (18 <= form_data["Age"] <= 65):
        return render_template(
            "error.html",
            message="Age must be between 18 and 65."
        )

    if not (1 <= form_data["Tenure"] <= 60):
        return render_template(
            "error.html",
            message="Tenure must be between 1 and 60."
        )

    if not (1 <= form_data["Usage Frequency"] <= 30):
        return render_template(
            "error.html",
            message="Usage Frequency must be between 1 and 30."
        )

    if not (0 <= form_data["Support Calls"] <= 10):
        return render_template(
            "error.html",
            message="Support Calls must be between 0 and 10."
        )

    if not (0 <= form_data["Payment Delay"] <= 30):
        return render_template(
            "error.html",
            message="Payment Delay must be between 0 and 30."
        )

    if not (100 <= form_data["Total Spend"] <= 1000):
        return render_template(
            "error.html",
            message="Total Spend must be between 100 and 1000."
        )

    if not (1 <= form_data["Last Interaction"] <= 30):
        return render_template(
            "error.html",
            message="Last Interaction must be between 1 and 30."
        )

    # ==========================================================
    # Prediction
    # ==========================================================

    prediction_result, probability = controller.predict(
        form_data
    )

    # ==========================================================
    # Return Result
    # ==========================================================

    return render_template(

        "result.html",

        prediction=prediction_result,

        probability=round(
            probability * 100,
            2
        )

    )