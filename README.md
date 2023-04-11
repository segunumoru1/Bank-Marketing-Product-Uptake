# Bank-Marketing-Product-Uptake
The binary classification goal is to predict if the client will subscribe a term deposit (variable y).

## Problem Statement
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed.

## Import necessary libraries and load file

load and read the dataset using pd

## Features of the Dataset
1 - age (numeric)

2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services")

3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)

4 - education (categorical: "unknown","secondary","primary","tertiary")

5 - default: has credit in default? (binary: "yes","no")

6 - balance: average yearly balance, in euros (numeric)

7 - housing: has housing loan? (binary: "yes","no")

8 - loan: has personal loan? (binary: "yes","no")

9 - contact: contact communication type (categorical: "unknown","telephone","cellular")

10 - day: last contact day of the month (numeric)

11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")

12 - duration: last contact duration, in seconds (numeric)

13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)

15 - previous: number of contacts performed before this campaign and for this client (numeric)

16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

Output variable (desired target):

17 - y - has the client subscribed a term deposit? (binary: "yes","no")

## Data Inspection and Manipulation
1. shape of the data
2. information of the data
3. checking null values
4. checking the columns of the dataset
5. descriptive statistical analysis

## Exploratory Data Analysis And Visualization
- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis

### Univariate Analysis
basically, looking at one feature and its visualization

### Bivariate Analysis
basically, comparing two features and its visualization

### Multivariate Analysis
it is comparing of three or more features and its visualization

## Preparing Models for Machine Learning
it is importing from scikitlearn the necessary algorithms for machine learning. The train_test_split were imported from sklearn on a test size of 0.3 that is 70:30 ratio for training and testing of the dataset

## One-Hot Encoding
With one-hot encoding, we can convert categorical feature into numerical. Each value of a column is pivoted into a column of its own. The values in this new column will be either 1 or 0 to show whether that value exist or not.

## Training the model
The following Algorithm were used to train and test the dataset:
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbor Classifier
- Support Vector Machine 
- XGBoost Classifier
- Logistic Regression

#### Decision Tree Classifier
the accuracy score for Decision Tree Classifier is approximately 84.8%

#### Random Forest Classifier
the accuracy score for Random Forest Classifier is approximately 90%

#### K-Nearest Neighbor Classifier
the accuracy score for K-Nearest Neighbor Classifier is approximately 86.6%

#### Support Vector Machine
the accuracy score for Support Vector Classifier is approximately 90.5%

#### XGBoost Classifier
the accuracy score for XGBoost Classifier is approximately 89.0%

#### Linear Regression Classifier
the accuracy score for Logistic Regression is approximately 88.5%

## Preparing the Models for training and validation
Confusion Matrix and cross validation score report was done using the below by looping through all the models
- from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
- from sklearn.metrics import accuracy_score, classification_report

### Observation on Metric Evalaution from the Confusion Matrix
it is observed that XGBoost has accuracy score of 90.6% with a high recall of about 0.6 and 0.42 precision respectively compared to other models that was trained and validated. The implication of that is there is 0.6 likelihood that there will be a bank product uptake while 0.4 won't take a bank product based on the model performance in terms of recall, precision and accuracy. so 42% were predicted accurately, i.e were actually true while 60% shows that our model predicted accurately.


## Inspect Feature Importances
Feature importance refers to a technique that calculate a score for all imput features for a given model- the scores simply reperesent the importance of each feature. A higher score means that a specific feature will have a larger effect on the model that has been used to predict our target variable.

#### Observation from feature importance
It is observed that duration and previous class contribute a larger percentage of about 0.048 and 0.032 respectively to the prediction of our target variable on a XGBoost Classifier model that has been trained and validated.

### A Bit of Feature Selection
XGB_importances[: 7].sort_values(by = 'Importance').index.values
array(['campaign', 'balance', 'age', 'day', 'pdays', 'previous',
       'duration'], dtype=object)
# Create a new X train with only 5 features
X_train2 = X_train[['campaign', 'balance', 'age', 'day', 'pdays', 'previous',
       'duration']]
X_train2.head()
campaign	balance	age	day	pdays	previous	duration
3257	1	1588	31	3	-1	0	14
2507	6	2613	42	30	-1	0	174
3701	1	5108	51	8	102	8	272
4287	14	3337	50	31	-1	0	24
945	3	1066	41	11	-1	0	109

Create a new X_valid with only 6 features so we can predict on them
X_valid2 = X_valid[['campaign', 'balance', 'age', 'day', 'pdays', 'previous',
       'duration']]

Train and predict
xgb_clf.fit(X_train2, y_train)
pred2 = xgb_clf.predict(X_valid2)
​
Print accuracy score
print(accuracy_score(pred2, y_valid))
0.8975681650700074

### Observation
it is observed that XGBoost has an accuracy of approximately 89.8% based on the feature importance and selection done in other to see those that contributed most to the model performance and its accuracy when retrained with a training set of the selected features.

## Evaluation with K-Fold Cross Validation
cross validation, which is sometimes called rotation estimation or out-of-sample testing, is a model validation technique for assessing how the results of a statistical analysis or model performance will fit into an unseen dataset. The main purpose is to to prevent overfitting; which occurs when a model is trained too well on the training dataset and performs poorly on a new, unseen dataset.

## Summary on K-fold Cross Validation
it is seen that the XGBoost model has predicted well and can generalize on a new data set based on the average score of 89.3% accuracy and a recall of 37% compared to the other model trained. The model is ready for deployment into an API which Heroku can be used as our deployment platform.
