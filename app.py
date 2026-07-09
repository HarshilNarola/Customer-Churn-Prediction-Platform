from flask import Flask

from config.settings import Config

from routes.about_routes import about
from routes.batch_prediction_routes import batch_prediction
from routes.main_routes import main
from routes.model_routes import model
from routes.prediction_routes import prediction


def create_app() -> Flask:
    """
    Create and configure the Flask application.
    """

    app = Flask(__name__)

    app.config.from_object(Config)

    # ==========================================================
    # Register Blueprints
    # ==========================================================

    app.register_blueprint(main)

    app.register_blueprint(prediction)

    app.register_blueprint(batch_prediction)

    app.register_blueprint(model)

    app.register_blueprint(about)

    return app


app = create_app()


if __name__ == "__main__":

    app.run(

        debug=app.config.get("DEBUG", False)

    )