# Bank Marketing Product Uptake

## Problem Statement

The objective of this project is to predict whether a client will subscribe to a term deposit (target variable 'y') based on the data related to direct marketing campaigns conducted by a Portuguese banking institution. These campaigns involved phone calls to clients, and multiple contacts were often necessary to determine if the client would subscribe to the bank's term deposit.

## Table of Contents
1. [Importing Libraries and Loading Data](#importing-libraries-and-loading-data)
2. [Dataset Features](#dataset-features)
3. [Data Inspection and Manipulation](#data-inspection-and-manipulation)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-and-visualization)
5. [Preparing Models for Machine Learning](#preparing-models-for-machine-learning)
6. [One-Hot Encoding](#one-hot-encoding)
7. [Model Training](#training-the-model)
   - Decision Tree Classifier
   - Random Forest Classifier
   - K-Nearest Neighbor Classifier
   - Support Vector Machine 
   - XGBoost Classifier
   - Logistic Regression
8. [Model Evaluation](#preparing-the-models-for-training-and-validation)
   - Confusion Matrix
   - Cross-Validation Score
9. [Feature Importance](#inspect-feature-importances)
10. [Feature Selection](#a-bit-of-feature-selection)
11. [K-Fold Cross Validation](#evaluation-with-k-fold-cross-validation)
12. [Summary and Deployment](#summary-on-k-fold-cross-validation)

## Importing Libraries and Loading Data
- Numpy
- Pandas
- Matplotlib 
- Seaborn
- Scikit-learn
- Scipy

To begin, we import the necessary Python libraries and load the dataset using Pandas.

## Dataset Features

1. Age (numeric)
2. Job: Type of job (categorical)
3. Marital status (categorical)
4. Education (categorical)
5. Default: Has credit in default? (binary)
6. Balance: Average yearly balance, in euros (numeric)
7. Housing: Has a housing loan? (binary)
8. Loan: Has a personal loan? (binary)
9. Contact: Contact communication type (categorical)
10. Day: Last contact day of the month (numeric)
11. Month: Last contact month of the year (categorical)
12. Duration: Last contact duration, in seconds (numeric)
13. Campaign: Number of contacts performed during this campaign (numeric)
14. Pdays: Number of days since the client was last contacted (numeric)
15. Previous: Number of contacts performed before this campaign (numeric)
16. Poutcome: Outcome of the previous marketing campaign (categorical)
17. Output variable (target): Has the client subscribed to a term deposit? (binary)

## Data Inspection and Manipulation

Shape of the data
Information about the data
Checking for null values
Checking the columns of the dataset
Descriptive statistical analysis

## Exploratory Data Analysis and Visualization

- **Univariate Analysis**: We explore individual features and visualize them.
- **Bivariate Analysis**: We compare pairs of features and visualize relationships.
- **Multivariate Analysis**: We analyze interactions between three or more features.

## Preparing Models for Machine Learning

We import machine learning algorithms from scikit-learn and split the data into training and testing sets using the train_test_split method (70:30 ratio).

## One-Hot Encoding

We apply one-hot encoding to convert categorical features into numerical ones, allowing us to use them in machine learning models.

## Model Training

We train various machine learning algorithms, including Decision Tree, Random Forest, K-Nearest Neighbor, Support Vector Machine, XGBoost, and Logistic Regression, to predict the term deposit subscription.

### Model Evaluation

We evaluate the models using metrics such as accuracy, precision, recall, and F1-score. Notably, XGBoost performs well with an accuracy of approximately 90.6%.

## Feature Importance

We inspect feature importances and find that 'duration' and 'previous' play significant roles in predicting the target variable.

## Feature Selection

We experiment with feature selection, creating a new dataset with selected features. The XGBoost model maintains an accuracy of about 89.8% with these selected features.

## K-Fold Cross Validation

We perform K-fold cross-validation to assess model generalization and avoid overfitting. XGBoost shows an average accuracy of 89.3% and a recall of 37%, indicating its suitability for deployment.

## Summary and Deployment

In summary, this project aims to predict term deposit subscriptions based on direct marketing campaign data. The XGBoost model demonstrates promising performance and is ready for deployment, possibly through an API using platforms like Heroku.