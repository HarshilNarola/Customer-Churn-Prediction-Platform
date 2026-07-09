import os

import gdown


MODEL_PATH = "models/calibrated_best_model.pkl"

FILE_ID = os.environ.get("MODEL_FILE_ID")


def download_model():
    """
    Download the calibrated model from Google Drive
    if it does not already exist.
    """

    if os.path.exists(MODEL_PATH):

        print("Calibrated model already exists.")

        return

    if not FILE_ID:

        raise ValueError(

            "Environment variable 'MODEL_FILE_ID' is not set."

        )

    os.makedirs("models", exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print("Downloading calibrated model...")

    success = gdown.download(

        url=url,

        output=MODEL_PATH,

        quiet=False

    )

    if success is None or not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(

            "Failed to download calibrated model."

        )

    print("Calibrated model downloaded successfully.")