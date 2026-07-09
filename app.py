import os

from utils.model_downloader import download_model

# ==========================================================
# Download Calibrated Model
# ==========================================================

download_model()

from flask import Flask

from config.settings import Config

from routes.about_routes import about
from routes.batch_prediction_routes import batch_prediction
from routes.main_routes import main
from routes.model_routes import model
from routes.prediction_routes import prediction


# ==========================================================
# Create Flask Application
# ==========================================================

def create_app() -> Flask:
    """
    Create and configure the Flask application.
    """

    app = Flask(__name__)

    app.config.from_object(Config)

    # ------------------------------------------------------
    # Register Blueprints
    # ------------------------------------------------------

    app.register_blueprint(main)

    app.register_blueprint(prediction)

    app.register_blueprint(batch_prediction)

    app.register_blueprint(model)

    app.register_blueprint(about)

    return app


app = create_app()


# ==========================================================
# Run Application
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(os.environ.get("PORT", 5000)),

        debug=False

    )