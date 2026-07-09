from services.prediction_service import PredictionService


class PredictionController:
    """
    Controller for handling single customer
    churn prediction requests.
    """

    def __init__(self) -> None:

        self.prediction_service = PredictionService()

    def predict(
        self,
        form_data: dict
    ):
        """
        Predict customer churn.
        """

        return self.prediction_service.predict(
            form_data
        )