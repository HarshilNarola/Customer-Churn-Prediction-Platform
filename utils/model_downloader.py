import os

import gdown


MODEL_PATH = "models/calibrated_best_model.pkl"

FILE_ID = "1vGDptoJyUaanbgBRhFTNZ4tfEeQ43en3"


def download_model():

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

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Failed to download calibrated model."
        )

    print("Calibrated model downloaded successfully.")