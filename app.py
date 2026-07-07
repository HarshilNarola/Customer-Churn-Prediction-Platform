from flask import Flask

from config.settings import Config

from routes.main_routes import main
from routes.prediction_routes import prediction
from routes.dashboard_routes import dashboard
from routes.model_routes import model
from routes.about_routes import about
from routes.batch_prediction_routes import batch_prediction

def create_app():

    app = Flask(__name__)

    app.config.from_object(Config)

    app.register_blueprint(main)

    app.register_blueprint(prediction)

    app.register_blueprint(dashboard)

    app.register_blueprint(model)

    app.register_blueprint(about)
    
    app.register_blueprint(batch_prediction)
    
    return app


app = create_app()


if __name__ == "__main__":

    app.run(
        debug=True
    )