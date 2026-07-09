"""
Application configuration settings.
"""

import os


class Config:
    """
    Base configuration for the Flask application.
    """

    # ==========================================================
    # Flask Configuration
    # ==========================================================

    SECRET_KEY = "customer-churn-secret-key"

    DEBUG = True

    TESTING = False

    # ==========================================================
    # Project Directories
    # ==========================================================

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

    TEMPLATE_FOLDER = os.path.join(
        BASE_DIR,
        "templates"
    )

    UPLOAD_FOLDER = os.path.join(
        BASE_DIR,
        "uploads"
    )

    DOWNLOAD_FOLDER = os.path.join(
        BASE_DIR,
        "downloads"
    )