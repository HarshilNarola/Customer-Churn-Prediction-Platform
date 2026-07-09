import os
import gdown


MODEL_PATH = "models/calibrated_best_model.pkl"

FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"


def download_model():

    if os.path.exists(MODEL_PATH):
        return

    os.makedirs("models", exist_ok=True)

    url = f"https://drive.google.com/file/d/1vGDptoJyUaanbgBRhFTNZ4tfEeQ43en3/view?usp=drive_link"

    print("Downloading calibrated model...")

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

    print("Download completed.")