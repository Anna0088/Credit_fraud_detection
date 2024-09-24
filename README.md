
# Credit Card Fraud Detection using Logistic Regression

This project aims to detect fraudulent credit card transactions using Logistic Regression. By analyzing transaction data, we develop a predictive model that can distinguish between legitimate and fraudulent transactions, which is crucial for financial institutions to prevent losses due to fraud.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Prerequisites](#prerequisites)
- [Code Explanation](#code-explanation)
  - [Importing Libraries](#importing-libraries)
  - [Loading the Dataset](#loading-the-dataset)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Data Preprocessing](#data-preprocessing)
  - [Balancing the Dataset](#balancing-the-dataset)
  - [Feature Selection and Data Splitting](#feature-selection-and-data-splitting)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Results and Interpretation](#results-and-interpretation)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Credit card fraud is a significant issue in the financial sector, leading to substantial financial losses annually. With the increase in online transactions, it's imperative to develop systems that can detect and prevent fraudulent activities efficiently. This project utilizes Logistic Regression, a statistical method for binary classification, to identify fraudulent transactions.

## Dataset Description

The dataset used is the **Credit Card Fraud Detection** dataset available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders in September 2013.

- **Features**:
  - The dataset consists of numerical input variables, which are the result of a PCA transformation (due to confidentiality issues).
  - **Time**: Seconds elapsed between each transaction and the first transaction.
  - **Amount**: Transaction amount.
- **Target Variable**:
  - **Class**: Indicates if a transaction is fraudulent (1) or legitimate (0).

**Imbalance in Data**:

- Legitimate transactions: 99.827% of the dataset.
- Fraudulent transactions: 0.173% of the dataset.

## Prerequisites

- Python 3.x
- Libraries:
  - NumPy
  - Pandas
  - Scikit-learn

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn
```

## Code Explanation

### Importing Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning algorithms and evaluation metrics.

### Loading the Dataset

```python
# Loading the dataset into a Pandas DataFrame
credit_card_data = pd.read_csv('/content/credit_data.csv')
```

- The dataset is loaded into a DataFrame named `credit_card_data`.

### Exploratory Data Analysis (EDA)

#### Viewing the Dataset

```python
# First 5 rows
credit_card_data.head()

# Last 5 rows
credit_card_data.tail()
```

- **head()** and **tail()** functions give a glimpse of the dataset.

#### Dataset Information

```python
# Dataset information
credit_card_data.info()
```

- Provides information about the data types and non-null values in each column.

#### Checking for Missing Values

```python
# Checking for missing values
credit_card_data.isnull().sum()
```

- Ensures that there are no missing values in the dataset.

#### Class Distribution

```python
# Distribution of legitimate and fraudulent transactions
credit_card_data['Class'].value_counts()
```

- **Output**:
  - `0`: 284315 (legitimate transactions)
  - `1`: 492 (fraudulent transactions)
- Highlights the class imbalance in the dataset.

### Data Preprocessing

#### Separating Legitimate and Fraudulent Transactions

```python
# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)
```

- **legit**: DataFrame containing legitimate transactions.
- **fraud**: DataFrame containing fraudulent transactions.
- **Shapes**:
  - Legitimate: `(284315, 31)`
  - Fraudulent: `(492, 31)`

#### Statistical Measures

```python
# Statistical measures of legitimate transactions
legit.Amount.describe()

# Statistical measures of fraudulent transactions
fraud.Amount.describe()
```

- Provides insights into the transaction amounts for both classes.

#### Comparing Transactions

```python
# Comparing mean values of all features for both classes
credit_card_data.groupby('Class').mean()
```

- Helps identify patterns or differences between legitimate and fraudulent transactions.

### Balancing the Dataset

Due to the significant class imbalance, the model may become biased towards predicting the majority class. To address this, we balance the dataset.

#### Under-Sampling Legitimate Transactions

```python
# Under-sampling legitimate transactions
legit_sample = legit.sample(n=492)
```

- Randomly selects 492 legitimate transactions to match the number of fraudulent transactions.

#### Creating a New Dataset

```python
# Concatenating the fraudulent transactions with the under-sampled legitimate transactions
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Shuffling the dataset
new_dataset = new_dataset.sample(frac=1).reset_index(drop=True)
```

- **new_dataset** now contains an equal number of legitimate and fraudulent transactions.

#### Verifying the New Dataset

```python
# Class distribution in the new dataset
new_dataset['Class'].value_counts()

# Output:
# 0    492
# 1    492
```

- Ensures that the dataset is balanced.

#### Statistical Measures of the New Dataset

```python
# Comparing mean values in the new dataset
new_dataset.groupby('Class').mean()
```

- Checks if the features still have distinguishing characteristics between the classes.

### Feature Selection and Data Splitting

#### Separating Features and Target

```python
# Separating the data into features (X) and target (Y)
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
```

- **X**: Features used for prediction.
- **Y**: Target variable.

#### Splitting the Data

```python
# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)
```

- **train_test_split**:
  - `test_size=0.2`: 20% of the data is used for testing.
  - `stratify=Y`: Ensures that both classes are equally represented in the train and test sets.
  - `random_state=2`: For reproducibility.
- **Output Shapes**:
  - X: `(984, 30)`
  - X_train: `(787, 30)`
  - X_test: `(197, 30)`

### Model Training

#### Initializing Logistic Regression Model

```python
# Initializing the Logistic Regression model
model = LogisticRegression()
```

#### Training the Model

```python
# Training the model with the training data
model.fit(X_train, Y_train)
```

- The model learns the relationship between features and the target variable.

### Model Evaluation

#### Evaluating on Training Data

```python
# Predicting on training data
X_train_prediction = model.predict(X_train)

# Calculating accuracy on training data
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data:', training_data_accuracy)
```

- **Accuracy**: Proportion of correctly predicted instances out of all instances.

#### Evaluating on Test Data

```python
# Predicting on test data
X_test_prediction = model.predict(X_test)

# Calculating accuracy on test data
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data:', test_data_accuracy)
```

- Evaluates how well the model generalizes to unseen data.

## Results and Interpretation

- **Accuracy on Training Data**: Approximately 93%
- **Accuracy on Test Data**: Approximately 93%

The model achieves high accuracy on both training and test data, indicating good performance. However, accuracy alone may not be sufficient due to the nature of the problem.

**Note**: In fraud detection, false negatives (fraudulent transactions classified as legitimate) are more critical than false positives. Therefore, additional evaluation metrics such as precision, recall, and F1-score should be considered for a comprehensive assessment.

## Conclusion

This project demonstrates how to build a logistic regression model to detect credit card fraud. By balancing the dataset and carefully preprocessing the data, the model achieves high accuracy. However, in real-world applications, further steps such as cross-validation, hyperparameter tuning, and using more sophisticated algorithms might be necessary to improve performance.

## References

- [Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Logistic Regression - Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Imbalanced Classification - Machine Learning Mastery](https://machinelearningmastery.com/what-is-imbalanced-classification/)
- [Evaluation Metrics for Classification Models](https://towardsdatascience.com/evaluation-metrics-for-classification-models-7f4b44a0b81a)
- [Under-sampling Techniques](https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets)
