
#Comprehensive Analysis and Predictive Modeling of Employee Attrition at Salifort Motors         

## Overview

This project aims to analyze employee data from Salifort Motors to predict employee retention and provide actionable insights to improve employee satisfaction and retention rates. Various machine learning models were implemented, including logistic regression and tree-based methods, with a focus on improving performance through feature engineering.


## Introduction

The goal of this project is to predict whether employees will leave the company and identify key factors influencing employee retention. By analyzing the provided dataset, we aim to gain insights into employee behavior and propose recommendations to improve retention rates.

## Dataset

The dataset contains various features related to employee performance, satisfaction, and engagement, including:
- Number of projects
- Average monthly hours
- Last evaluation score
- Satisfaction level
- Whether the employee left the company (target variable)

## Project Steps

### 1. Data Import and Preprocessing

- Load the dataset using pandas.
- Inspect the data, handle missing values and duplicates.
- Perform initial exploratory data analysis.

### 2. Exploratory Data Analysis (EDA)

- Visualize distributions using histograms and boxplots.
- Analyze correlations between features using a correlation matrix and pair plots.

### 3. Feature Engineering

- Create new features and transform existing ones.
- Encode categorical variables and scale numerical features.

### 4. Model Building

- Split the data into training and testing sets.
- Train various machine learning models, including logistic regression and tree-based models.
- Perform hyperparameter tuning using GridSearchCV.

### 5. Model Evaluation

- Evaluate models using metrics such as AUC, precision, recall, F1-score, and accuracy.
- Visualize the confusion matrix to understand model performance.

### 6. Saving and Loading the Model

- Define functions to save and load models using `pickle`.
- Save the best model to disk and demonstrate loading it back for predictions.

### 7. Conclusion and Recommendations

- Summarize key findings and model results.
- Provide actionable recommendations to improve employee retention.
- Suggest next steps for further analysis.

## Model Results Summary

### Logistic Regression

- Precision: 80%
- Recall: 83%
- F1-Score: 80%
- Accuracy: 83%

### Tree-based Machine Learning

- **Decision Tree Model**:
  - AUC: 93.8%
  - Precision: 87.0%
  - Recall: 90.4%
  - F1-Score: 88.7%
  - Accuracy: 96.2%
- Random Forest Model: Modestly outperformed the decision tree model.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pickle

