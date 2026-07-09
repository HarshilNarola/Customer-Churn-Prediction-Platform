from flask import (
    Blueprint,
    render_template,
    request
)

from controllers.prediction_controller import PredictionController


prediction = Blueprint(
    "prediction",
    __name__
)

controller = PredictionController()

VALIDATION_RULES = {

    "Age": (18, 65),

    "Tenure": (1, 60),

    "Usage Frequency": (1, 30),

    "Support Calls": (0, 10),

    "Payment Delay": (0, 30),

    "Total Spend": (100, 1000),

    "Last Interaction": (1, 30)

}


@prediction.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():
    """
    Handle single customer churn prediction.
    """

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

    try:

        form_data = {

            "Age": int(request.form["Age"]),

            "Gender": request.form["Gender"],

            "Tenure": int(request.form["Tenure"]),

            "Usage Frequency": int(
                request.form["Usage Frequency"]
            ),

            "Support Calls": int(
                request.form["Support Calls"]
            ),

            "Payment Delay": int(
                request.form["Payment Delay"]
            ),

            "Subscription Type": request.form[
                "Subscription Type"
            ],

            "Contract Length": request.form[
                "Contract Length"
            ],

            "Total Spend": int(
                request.form["Total Spend"]
            ),

            "Last Interaction": int(
                request.form["Last Interaction"]
            )

        }

    except (KeyError, ValueError):

        return render_template(

            "error.html",

            message="Invalid form data submitted."

        )

    # ==========================================================
    # Input Validation
    # ==========================================================

    for field, (minimum, maximum) in VALIDATION_RULES.items():

        value = form_data[field]

        if not minimum <= value <= maximum:

            return render_template(

                "error.html",

                message=(
                    f"{field} must be between "
                    f"{minimum} and {maximum}."
                )

            )

    # ==========================================================
    # Prediction
    # ==========================================================

    try:

        prediction_result, probability = controller.predict(
            form_data
        )

    except ValueError as error:

        return render_template(

            "error.html",

            message=str(error)

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