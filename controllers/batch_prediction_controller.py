from services.batch_prediction_service import BatchPredictionService


class BatchPredictionController:
    """
    Controller responsible for handling batch
    customer churn prediction requests.
    """

    def __init__(self) -> None:
        """
        Initialize the batch prediction controller.
        """

        self.batch_prediction_service = BatchPredictionService()

    # ==========================================================
    # Process Batch Prediction
    # ==========================================================

    def process(
        self,
        filepath: str
    ) -> tuple:
        """
        Process a CSV file containing multiple customers.

        Parameters
        ----------
        filepath : str
            Path to the uploaded CSV file.

        Returns
        -------
        tuple
            (output_file, None) if successful,
            (None, missing_columns) otherwise.
        """

        dataframe = self.batch_prediction_service.load_csv(
            filepath
        )

        missing_columns = self.batch_prediction_service.validate_columns(
            dataframe
        )

        if missing_columns:

            return None, missing_columns

        processed_dataframe = self.batch_prediction_service.preprocess(
            dataframe
        )

        predictions, probabilities = self.batch_prediction_service.predict(
            processed_dataframe
        )

        output_file = self.batch_prediction_service.save_results(

            dataframe,

            predictions,

            probabilities

        )

        return output_file, None