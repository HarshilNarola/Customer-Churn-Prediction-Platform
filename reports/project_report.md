# Customer Churn Prediction Platform

# Project Report

---

# Abstract

Customer retention is one of the most important challenges faced by subscription-based businesses. Acquiring new customers is often significantly more expensive than retaining existing ones, making customer churn prediction a valuable business application of machine learning.

This project presents the design and development of a **Customer Churn Prediction Platform**, an end-to-end machine learning web application capable of predicting whether a customer is likely to discontinue a service. The platform integrates data preprocessing, exploratory data analysis (EDA), model training, evaluation, hyperparameter tuning, probability calibration, and deployment through a Flask-based web application.

Multiple supervised machine learning algorithms were implemented and evaluated, including Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine. Their performance was compared using standard classification metrics such as Accuracy, Precision, Recall, F1 Score, and ROC-AUC. Five-fold cross validation and GridSearchCV were employed to improve model robustness and optimize hyperparameters. Finally, probability calibration using the sigmoid method was applied to generate reliable probability estimates.

The developed web application supports both single customer prediction through an interactive form and batch prediction using CSV file uploads. The platform also provides model comparison reports, exploratory data analysis visualizations, and downloadable prediction results, making it suitable for both business users and educational purposes.

---

# Table of Contents

1. Introduction
2. Problem Statement
3. Project Objectives
4. System Overview
5. Dataset Description
6. Exploratory Data Analysis
7. Data Preprocessing
8. Machine Learning Models
9. Model Evaluation
10. Cross Validation
11. Hyperparameter Tuning
12. Probability Calibration
13. Final Model Selection
14. Flask Web Application
15. Project Structure
16. Results
17. Conclusion
18. Future Improvements

---

# 1. Introduction

Customer churn prediction is a binary classification problem that aims to identify customers who are likely to discontinue using a company's products or services. Predicting churn enables organizations to take proactive measures, improve customer satisfaction, and reduce revenue loss through targeted retention strategies.

Advancements in machine learning have enabled organizations to analyze historical customer behaviour and accurately estimate churn probabilities. By combining customer demographics, subscription information, spending behaviour, service usage, and customer support interactions, predictive models can identify patterns associated with customer attrition.

This project implements a complete machine learning pipeline for customer churn prediction. Beginning with data inspection and exploratory data analysis, the project progresses through preprocessing, model training, cross validation, hyperparameter tuning, probability calibration, and deployment as a Flask web application. The resulting platform provides an intuitive interface for predicting churn for individual customers as well as large batches of customer records.

---

# 2. Problem Statement

Subscription-based businesses continuously face the challenge of losing customers due to changing customer preferences, competition, pricing strategies, or service dissatisfaction. Identifying customers who are likely to churn before they leave enables organizations to implement effective retention strategies and improve long-term customer relationships.

Traditional rule-based methods often fail to capture complex relationships among customer attributes. Therefore, machine learning techniques are employed to learn patterns from historical customer data and generate accurate churn predictions. The objective of this project is to develop a reliable, scalable, and user-friendly prediction platform capable of assisting organizations in identifying high-risk customers.

---

# 3. Project Objectives

The primary objectives of this project are:

- Develop an end-to-end customer churn prediction system.
- Perform comprehensive exploratory data analysis.
- Apply appropriate data preprocessing techniques.
- Train and compare multiple supervised machine learning algorithms.
- Evaluate models using standard classification metrics.
- Improve model performance through cross validation and hyperparameter tuning.
- Calibrate prediction probabilities for improved reliability.
- Deploy the final model using a Flask web application.
- Support both single customer prediction and batch prediction using CSV files.
- Provide reports and visualizations for better model interpretation.

---

# 4. System Overview

The Customer Churn Prediction Platform is a web-based machine learning application designed to predict customer churn using historical customer data. The platform integrates data preprocessing, model training, evaluation, and deployment into a single workflow.

The application allows users to perform predictions for both individual customers and multiple customers simultaneously through CSV file uploads. In addition to prediction, the platform provides machine learning reports, model comparison results, and exploratory data analysis visualizations.

## System Workflow

```text
Customer Dataset
        │
        ▼
 Data Inspection
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Preprocessing
        │
        ▼
Machine Learning Models
        │
        ▼
Cross Validation
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Probability Calibration
        │
        ▼
 Flask Web Application
        │
        ▼
 Customer Churn Prediction
```

## Main Components

The system consists of the following major components:

- Data Loading
- Data Inspection
- Exploratory Data Analysis
- Data Preprocessing
- Machine Learning Model Training
- Model Evaluation
- Cross Validation
- Hyperparameter Tuning
- Probability Calibration
- Flask Web Application
- Batch Prediction System

---

# 5. Dataset Description

The project uses a customer churn dataset containing customer demographic information, subscription details, service usage, payment behaviour, and churn status.

Each record represents one customer, while the target variable indicates whether the customer discontinued the service.

## Dataset Summary

| Property | Value |
|-----------|------:|
| Total Records | **64,374** |
| Total Features | **12** |
| Numerical Features | **9** |
| Categorical Features | **3** |
| Missing Values | **0** |
| Duplicate Records | **0** |
| Target Variable | **Churn** |

## Dataset Features

| Feature | Description |
|----------|-------------|
| CustomerID | Unique customer identifier |
| Age | Customer age |
| Gender | Customer gender |
| Tenure | Months with the company |
| Usage Frequency | Monthly service usage |
| Support Calls | Number of customer support requests |
| Payment Delay | Average payment delay |
| Subscription Type | Customer subscription plan |
| Contract Length | Subscription duration |
| Total Spend | Total customer spending |
| Last Interaction | Days since last interaction |
| Churn | Target variable |

The dataset contains a balanced distribution of churned and retained customers, making it suitable for supervised binary classification.

---

# 6. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the characteristics of the dataset before model development. The analysis focused on identifying data quality issues, studying feature distributions, examining relationships between variables, and understanding customer churn behaviour.

The following analyses were performed:

- Dataset inspection
- Missing value analysis
- Duplicate record detection
- Target variable analysis
- Numerical feature analysis
- Categorical feature analysis
- Feature versus target analysis
- Correlation analysis

## Major Findings

- No missing values were detected.
- No duplicate records were present.
- The dataset is balanced.
- CustomerID does not contribute to prediction.
- Customer tenure, payment delay, support calls, and usage frequency have strong relationships with churn.
- No severe multicollinearity was observed among predictor variables.

Detailed analysis and visualizations are available in the separate **EDA Report**.

---

# 7. Data Preprocessing

Before training the machine learning models, the dataset underwent several preprocessing steps to improve model performance and ensure consistency.

## Preprocessing Pipeline

The following preprocessing operations were applied:

- Removal of CustomerID
- Label Encoding of categorical variables
- Standard Scaling of numerical variables
- Stratified Train-Test Split (80:20)

### Label Encoding

Categorical features were converted into numerical representations using Label Encoding.

The encoded features include:

- Gender
- Subscription Type
- Contract Length

### Feature Scaling

Numerical features were standardized using StandardScaler to ensure that all variables contribute equally during model training.

### Train-Test Split

The dataset was divided into:

- Training Set (80%)
- Testing Set (20%)

A stratified split was used to preserve the original class distribution.

---

# 8. Machine Learning Models

Six supervised machine learning algorithms were implemented and evaluated for customer churn prediction.

## Models Implemented

| Model | Purpose |
|--------|----------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Rule-based classification |
| Random Forest | Ensemble learning |
| K-Nearest Neighbors | Instance-based learning |
| Gaussian Naive Bayes | Probabilistic classification |
| Support Vector Machine | Margin-based classification |

Each model was trained using the preprocessed dataset and evaluated using standard classification metrics.

## Evaluation Metrics

The following metrics were used for performance comparison:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The evaluation process identified the most suitable model for deployment based on predictive performance and reliability.

---

# 9. Model Evaluation

The trained machine learning models were evaluated on the testing dataset to compare their predictive performance.

The following evaluation techniques were applied:

- Confusion Matrix
- Classification Report
- ROC Curve
- Feature Importance (Decision Tree and Random Forest)
- Model Comparison Table

The evaluation results indicate that ensemble learning methods outperform individual classifiers on this dataset.

A detailed comparison of all trained models is provided in the **Model Comparison Report**, while the final deployed model was selected based on overall predictive performance and probability estimation quality.

---

# 10. Cross Validation

To evaluate the robustness and generalization capability of the machine learning models, **5-Fold Stratified Cross Validation** was performed.

During this process, the dataset was divided into five equally sized subsets while preserving the original class distribution. Each subset was used once as the validation set while the remaining subsets were used for training. The final performance was calculated as the average across all folds.

## Benefits

- Reduces evaluation bias.
- Improves model reliability.
- Ensures stable performance across different data partitions.
- Minimizes the risk of overfitting.

The following metrics were averaged across all folds:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Cross validation confirmed that the selected models maintained consistent performance across different training and validation splits.

---

# 11. Hyperparameter Tuning

Several machine learning algorithms contain configurable parameters that significantly influence predictive performance. To determine the optimal parameter values, **GridSearchCV** was applied.

The following models were tuned:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

GridSearchCV systematically evaluated multiple parameter combinations using five-fold cross validation and selected the configuration with the highest average validation accuracy.

## Benefits

- Improves model accuracy.
- Reduces overfitting.
- Produces more robust models.
- Automates parameter selection.

The optimized parameters for each model are available in the generated tuning reports.

---

# 12. Probability Calibration

Many classification algorithms produce probabilities that are not well calibrated. Although a model may correctly classify observations, the predicted probabilities may not accurately represent the true likelihood of churn.

To improve probability estimation, the final deployed model was calibrated using **CalibratedClassifierCV** with the **Sigmoid** calibration method.

## Advantages

- Produces more reliable probability estimates.
- Improves confidence in predictions.
- Reduces Brier Score.
- Supports better business decision-making.

The calibrated probabilities are displayed within the web application for both single and batch predictions.

---

# 13. Final Model Selection

After evaluating all machine learning models, the **Random Forest Classifier** was selected as the final deployed model.

The model demonstrated superior predictive performance while maintaining strong generalization capability after cross validation and hyperparameter tuning.

The selected model was further calibrated using sigmoid probability calibration to improve probability estimation.

## Reasons for Selection

- Highest overall predictive performance.
- Strong Accuracy, Precision, Recall and F1 Score.
- High ROC-AUC.
- Robust performance across cross validation folds.
- Reliable calibrated probabilities.

The final trained model was serialized using **Joblib** and integrated into the Flask application for deployment.

---

# 14. Flask Web Application

A Flask web application was developed to make the trained machine learning model accessible through an intuitive graphical user interface.

The application supports both individual customer prediction and batch prediction using CSV files.

## Application Features

- Home page
- Single customer prediction
- Batch prediction
- Model comparison page
- About page
- Error handling
- Downloadable prediction reports

### Single Prediction

Users enter customer information through an interactive web form. The application preprocesses the input, performs prediction using the calibrated Random Forest model, and displays both the predicted class and the associated probability.

### Batch Prediction

Users upload a CSV file containing multiple customer records. The application validates the uploaded file, preprocesses the data, performs predictions for all customers, and generates a downloadable CSV file containing the prediction results.

---

# 15. Project Structure

The project follows a modular architecture to improve maintainability and scalability.

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
├── templates/
├── tests/
├── uploads/
├── utils/
│
├── app.py
├── requirements.txt
└── README.md
```

The project separates responsibilities into independent modules such as services, controllers, routes, utilities, templates, and static resources, resulting in a clean and maintainable codebase.

---

# 16. Results

The completed platform successfully predicts customer churn using machine learning techniques integrated into a Flask web application.

The project achieved the following outcomes:

- Successful implementation of six supervised learning algorithms.
- Comprehensive exploratory data analysis.
- Effective preprocessing pipeline.
- Reliable model evaluation.
- Hyperparameter optimization using GridSearchCV.
- Stable performance through five-fold cross validation.
- Improved probability estimation through calibration.
- Functional web application supporting both single and batch prediction.

The final deployed model provides accurate churn predictions together with calibrated probability estimates suitable for practical decision-making.

---

# 17. Conclusion

This project successfully demonstrates the complete development of an end-to-end Customer Churn Prediction Platform using machine learning and Flask. The project began with data inspection and exploratory data analysis, followed by preprocessing, model training, evaluation, cross validation, hyperparameter tuning, probability calibration, and finally deployment as an interactive web application.

Six supervised machine learning algorithms were implemented and compared using multiple evaluation metrics, including Accuracy, Precision, Recall, F1 Score, and ROC-AUC. The Random Forest Classifier achieved the best overall performance and was selected as the final deployed model. To improve the reliability of predicted probabilities, the model was calibrated using the sigmoid method before deployment.

The Flask application provides an intuitive interface for both single customer prediction and batch prediction through CSV file uploads. In addition, the project includes comprehensive documentation, automated testing, exploratory data analysis reports, and model comparison reports, making it suitable for educational purposes as well as practical business applications.

Overall, the Customer Churn Prediction Platform demonstrates the successful integration of data science, machine learning, and web development into a modular, maintainable, and user-friendly application capable of supporting customer retention strategies through predictive analytics.

---


# References

1. Pedregosa, F., et al. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.

2. Flask Documentation  
   https://flask.palletsprojects.com/

3. Scikit-learn Documentation  
   https://scikit-learn.org/

4. Pandas Documentation  
   https://pandas.pydata.org/

5. Matplotlib Documentation  
   https://matplotlib.org/

6. Joblib Documentation  
   https://joblib.readthedocs.io/

---

# Appendix

The project repository contains the following supplementary resources:

- Exploratory Data Analysis Report
- Model Comparison Report
- Hyperparameter Tuning Summary
- Final Model Report
- Source Code
- Automated Test Scripts
- Trained Machine Learning Models
- Sample Dataset
- Sample Batch Prediction CSV