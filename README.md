# Customer Churn Prediction Platform

A full-stack Machine Learning web application that predicts whether a customer is likely to churn using supervised machine learning algorithms. The project demonstrates an end-to-end machine learning workflow, from data preprocessing and exploratory data analysis (EDA) to model training, evaluation, probability calibration, and deployment using Flask.

---

## Features

- Single customer churn prediction through a web form
- Batch prediction using CSV file upload
- Automatic data preprocessing
- Exploratory Data Analysis (EDA)
- Model comparison across multiple machine learning algorithms
- Hyperparameter tuning using GridSearchCV
- 5-Fold Cross Validation
- Probability calibration using Sigmoid Calibration
- Downloadable batch prediction results
- Performance reports and visualizations
- Responsive web interface built with Flask

---

## Technology Stack

### Backend

- Python
- Flask

### Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

### Data Visualization

- Matplotlib

### Frontend

- HTML5
- CSS3

### Development Tools

- Git
- GitHub
- VS Code

---

## Machine Learning Pipeline

```text
Dataset
    ↓
Data Inspection
    ↓
Data Leakage Analysis
    ↓
Data Preprocessing
    ↓
Exploratory Data Analysis
    ↓
Model Training
    ↓
Cross Validation
    ↓
Hyperparameter Tuning
    ↓
Model Evaluation
    ↓
Probability Calibration
    ↓
Flask Deployment
```

---

## Machine Learning Models

The following supervised learning algorithms were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine

The final deployed model is a **Calibrated Random Forest Classifier**.

---

## Project Structure

```text
Customer-Churn-Prediction-Platform/
│
├── config/
├── controllers/
├── data/
├── downloads/
├── models/
├── reports/
├── routes/
├── services/
├── static/
│   ├── css/
│   ├── images/
│   └── sample/
├── templates/
├── tests/
├── uploads/
├── utils/
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/HarshilNarola/Customer-Churn-Prediction-Platform.git
```

Move into the project directory.

```bash
cd Customer-Churn-Prediction-Platform
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate the virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask server.

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## Project Modules

- Data Loading
- Data Inspection
- Data Preprocessing
- Exploratory Data Analysis
- Model Training
- Cross Validation
- Hyperparameter Tuning
- Probability Calibration
- Model Evaluation
- Batch Prediction
- Single Prediction
- Flask Web Application

---

## Reports

The `reports/` directory contains:

- Model Comparison Report
- Final Model Report
- Hyperparameter Tuning Summary
- Grid Search Results
- EDA Report

---

## Test Scripts

The project includes test scripts for:

- Data Loader
- Data Inspector
- Data Leakage Detection
- Preprocessing Service
- Model Training Service
- Cross Validation
- Hyperparameter Tuning
- Evaluation Service
- Probability Calibration
- Final Model
- EDA Service

---

## Future Improvements

- Docker support
- User authentication
- Cloud deployment
- Explainable AI (SHAP/LIME)
- REST API
- Database integration

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for more information.