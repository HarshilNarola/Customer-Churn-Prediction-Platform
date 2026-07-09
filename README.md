# 📉 Customer Churn Prediction Platform

A Full Stack Machine Learning web application that predicts whether a customer is likely to churn using supervised machine learning algorithms.

The project demonstrates an end-to-end Machine Learning workflow including data preprocessing, exploratory data analysis (EDA), model training, evaluation, hyperparameter tuning, probability calibration, and deployment using Flask and Docker.

---

# 🌐 Live Demo

**Application**

https://customer-churn-prediction-platform-1-osgi.onrender.com

---

# ✨ Features

- Single Customer Churn Prediction
- Batch Prediction using CSV Upload
- Automatic Data Preprocessing
- Exploratory Data Analysis (EDA)
- Model Comparison
- Hyperparameter Tuning
- 5-Fold Cross Validation
- Probability Calibration
- Download Prediction Results
- Responsive User Interface
- Docker Deployment
- Render Cloud Deployment

---

# 🧠 Machine Learning Pipeline

```text
Customer Dataset
        │
        ▼
Data Inspection
        │
        ▼
Data Leakage Detection
        │
        ▼
Data Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Model Training
        │
        ▼
Cross Validation
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Probability Calibration
        │
        ▼
Flask Web Application
        │
        ▼
Docker Deployment
```

---

# 🤖 Machine Learning Models

The following classification algorithms were implemented and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

The final deployed model is a

**Calibrated Random Forest Classifier**

---

# 🛠️ Technology Stack

## Backend

- Python
- Flask

## Machine Learning

- Scikit-learn
- NumPy
- Pandas
- Joblib

## Data Visualization

- Matplotlib

## Frontend

- HTML5
- CSS3

## Deployment

- Docker
- Render

## Development Tools

- Git
- GitHub
- VS Code

---

# 📂 Project Structure

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
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/HarshilNarola/Customer-Churn-Prediction-Platform.git
```

Move into the project.

```bash
cd Customer-Churn-Prediction-Platform
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the application.

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 🐳 Docker Deployment

Build the Docker image.

```bash
docker build -t customer-churn-app .
```

Run the container.

```bash
docker run -p 5000:5000 customer-churn-app
```

Open

```
http://localhost:5000
```

---

# 📊 Reports

The project contains:

- Exploratory Data Analysis Report
- Cross Validation Report
- Final Model Report
- Project Report

---

# 🧪 Testing

The project includes test scripts for:

- Data Loader
- Data Inspector
- Data Leakage Detection
- Data Preprocessing
- Model Training
- Cross Validation
- Hyperparameter Tuning
- Probability Calibration
- Final Model Evaluation
- EDA Service

---

# 📷 Application Screenshots

You can add screenshots here.

Example:

- Home Page
- Prediction Page
- Result Page
- Batch Prediction
- About Page
- Models Page

---

# 🚀 Future Improvements

- User Authentication
- REST API
- Explainable AI (SHAP/LIME)
- Database Integration
- CI/CD Pipeline
- Kubernetes Deployment
- Monitoring Dashboard

---

# 👨‍💻 Author

**Harshil Narola**

GitHub

https://github.com/HarshilNarola

LinkedIn

(Add your LinkedIn profile here)

---

# ⭐ Project Status

Current Version

✅ Completed

- End-to-End Machine Learning Pipeline
- Flask Web Application
- Docker Deployment
- Render Deployment

---

## 📌 Note

This application automatically downloads the calibrated machine learning model when required during deployment.

The model file is not stored inside the GitHub repository because it exceeds GitHub's file size limit.
