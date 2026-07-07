import os


class Config:

    SECRET_KEY = "customer-churn-secret-key"

    BASE_DIR = os.path.abspath(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )

    MODEL_FOLDER = os.path.join(
        BASE_DIR,
        "models"
    )

    REPORT_FOLDER = os.path.join(
        BASE_DIR,
        "reports"
    )

    STATIC_FOLDER = os.path.join(
        BASE_DIR,
        "static"
    )