from services.batch_prediction_service import BatchPredictionService


class BatchPredictionController:

    def __init__(self):

        self.service = BatchPredictionService()

    def process(self, filepath):

        df = self.service.load_csv(filepath)

        missing = self.service.validate_columns(df)

        if missing:

            return None, missing

        processed_df = self.service.preprocess(df)

        prediction, probability = self.service.predict(processed_df)

        output = self.service.save_results(

            df,

            prediction,

            probability

        )

        return output, None