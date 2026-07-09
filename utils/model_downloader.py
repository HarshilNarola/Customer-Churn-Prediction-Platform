import os

import gdown


MODEL_PATH = "models/calibrated_best_model.pkl"

FILE_ID = "1vGDptoJyUaanbgBRhFTNZ4tfEeQ43en3"


def download_model():
    """
    Download the calibrated model from Google Drive
    if it does not already exist.
    """

    if os.path.exists(MODEL_PATH):

        print("Calibrated model already exists.")

        return

    os.makedirs("models", exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print("Downloading calibrated model...")

    gdown.download(

        url=url,

        output=MODEL_PATH,

        quiet=False

    )

    print("Model downloaded successfully.")