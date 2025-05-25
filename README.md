# Parkinson's Disease Prediction Project 🩺

This project focuses on developing a machine learning model to predict the presence of Parkinson's disease using a dataset of various voice measurements. The goal is to explore different classification algorithms and identify the most effective model for this task.

## 📋 Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Workflow](#workflow)
* [Installation](#installation)
* [Usage](#usage)
* [Model Training and Evaluation](#model-training-and-evaluation)
* [Performance Metrics](#performance-metrics)
* [Model Comparison](#model-comparison)
* [Results and Discussion](#results-and-discussion)
* [Contributing](#contributing)
* [License](#license)

## 🌟 Project Overview

Parkinson's disease is a progressive nervous system disorder that affects movement. Early diagnosis can significantly improve patient management. This project leverages voice measurements, as vocal impairments are common early symptoms, to build a predictive model. The process includes data cleaning, exploratory analysis, feature engineering, and model comparison.

## 💾 Dataset

The dataset used is `parkinson_disease.csv`, which contains various biomedical voice measurements from individuals with and without Parkinson's disease.

* **Features**: The dataset includes 23 attributes, primarily voice measures such as fundamental frequency variations, shimmer, jitter, and noise-to-harmonics ratios.
* **Target Variable**: `status` - 1 for Parkinson's, 0 for healthy.

The dataset was preprocessed to handle outliers and scaled using MinMaxScaler. Feature selection was performed using SelectKBest with chi-squared to select the top 30 features.
## ⚙️ Workflow

1.  **Data Loading and Initial Inspection**: Loaded the dataset and performed initial checks (shape, info, null values, duplicates).
2.  **Exploratory Data Analysis (EDA)**: Visualized data distributions. Investigated the target variable distribution.
3.  **Data Preprocessing**:
    * Separated features (X) and target (y).
    * Scaled numerical features using `MinMaxScaler`.
    * Selected the top 30 features using `SelectKBest` with the `chi2` test.
4.  **Train-Test Split**: Split the data into training (80%) and testing (20%) sets, stratifying by the target variable to maintain class proportions.
5.  **Handling Class Imbalance**: Applied SMOTE (Synthetic Minority Over-sampling Technique) **only on the training data** to address class imbalance.
6.  **Model Training**: Trained three different classification models:
    * Logistic Regression
    * XGBoost Classifier
    * Support Vector Classifier (SVC)
7.  **Model Evaluation**: Evaluated models using ROC AUC score, accuracy, classification reports, and confusion matrices on the test set.

## 📊 Model Training and Evaluation

Three models were trained and evaluated: Logistic Regression, XGBoost Classifier, and Support Vector Classifier (SVC).

### Performance Metrics

The primary metric for comparison was the **ROC AUC score** on the validation (test) set. Accuracy and Classification Reports were also considered.

| Model                       | Train ROC AUC Score | Validation ROC AUC Score | Validation Accuracy    |
| --------------------------- | ------------------- | ------------------------ | ---------------------- |
| Logistic Regression         | 0.8917              | 0.7996                   | (Refer to notebook)    |
| **XGBClassifier** | **1.0000** | **0.9190** | **(Refer to notebook)**|
| Support Vector Classifier (SVC) | 0.9332              | 0.8583                   | 0.8235 (from report)   |

### Model Comparison

* **XGBClassifier**: Achieved the **highest Validation ROC AUC score (0.9190)**, indicating the best predictive performance on unseen data among the tested models. However, a Train ROC AUC score of 1.0000 suggests it perfectly learned the training data, indicating significant **overfitting**.
* **Support Vector Classifier (SVC)**: Showed a **good balance** between training and validation performance (Train ROC AUC: 0.9332, Validation ROC AUC: 0.8583). The gap between train and validation scores is the smallest, suggesting it is **less overfit** compared to the other models.
* **Logistic Regression**: Had the lowest performance on the validation set and also showed signs of overfitting.

#### Classification Report - Support Vector Classifier (SVC)

The SVC model, while not the top performer by ROC AUC, showed good generalization:
```text
Classification Report - SVM:
              precision    recall  f1-score   support

         0.0       0.64      0.69      0.67        13
         1.0       0.89      0.87      0.88        38

    accuracy                           0.82        51
   macro avg       0.77      0.78      0.77        51
weighted avg       0.83      0.82      0.83        51
```
This report indicates an accuracy of approximately 82% for SVC on the test set. It performs well in identifying Parkinson's cases (class 1.0) with high precision and recall.

## 💡 Results and Discussion

The **XGBoost Classifier** demonstrated the strongest predictive capability on the test data, achieving a Validation ROC AUC of **0.9190**. This suggests it is very effective at distinguishing between healthy individuals and those with Parkinson's based on the selected voice features.

However, the perfect training score (1.0000) for XGBoost is a strong sign of **overfitting**. While it generalizes well to this specific test set, its performance might be less robust on entirely new, unseen data compared to a model with less overfitting.

The **Support Vector Classifier (SVC)**, with a Validation ROC AUC of **0.8583** and an accuracy of **82%**, presented a more balanced profile. It had the smallest difference between training and validation scores, indicating better generalization and less overfitting than XGBoost.

**Conclusion**:
For maximizing predictive accuracy on data similar to the test set, **XGBoost is the preferred model**. If robustness against overfitting is a higher priority, **SVC provides a more reliable and balanced alternative**. Further work could involve hyperparameter tuning for all models (especially XGBoost to reduce overfitting) and exploring other feature engineering techniques.
