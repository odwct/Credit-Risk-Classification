# Module 12 Credit-Risk-Classification

## Overview of the Analysis

The analysis was conducted using a supervised learning experiment for predicting loan statuses based on a provided dataset. The dataset contained financial information about loans, including Loan Size, Interest Rate, Borrower Income, Debt-to-Income Ratio, Number of Accounts, Derogatory Marks, and Total Debt. The goal was to determine if a loan was classified as healthy (0) or high risk (1).

The analysis consisted og the following stages:

1. Data Preparation: A label set called "y" was created from the "loan status" column, and a label called "X" was created for the remaining columns. The labels were checked by using the `value_counts` function and split into training and testing sets using the `train_test_split` function.

2. Model Selection: For classification tasks, we used a machine-learning algorithm. Specifically, we employed 'LogisticRegression' on the original dataset and 'RandomOverSampler' to rebalance the training data in our analysis.

3. Model Training: We split the data into training and validation sets, and trained the selected models on the training data.

4. Model Evaluation: We evaluated the models' performance using various metrics, including balanced accuracy, a confusion matrix, and a classification report showing precision, recall, and F1-score for healthy loans ("0") and high-risk loans ("1"). 

## Results

* Machine Learning Model 1: Logistic Regression (Original Data)
  * Confusion Matrix:

          Predicted 0    Predicted 1
Actual 0    18663         102
Actual 1    56            615

* True Positives (TP) for predicted "1" (high-risk loan): 615
* True Negatives (TN) for predicted "0" (healthy loan): 18663
* False Positives (FP) for predicted "1": 102
* False Negatives (FN) for predicted "0": 56


* Accuracy Score: 0.95
* Classification Report:

              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2: Logistic Regression (Oversampled Data)
  * Confusion Matrix:

          Predicted 0    Predicted 1
Actual 0    18649         116
Actual 1        4         615

* True Positives (TP) for predicted "1" (high-risk loan): 615
* True Negatives (TN) for predicted "0" (healthy loan): 18649
* False Positives (FP) for predicted "1": 116
* False Negatives (FN) for predicted "0": 4


* Accuracy Score: 0.99

* Classification Report
                                                            
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384

  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

To predict loan statuses, was conducted an analysis using two Logistic Regression models. One was trained on the original data while the other used oversampled data. Both models showed excellent performance in predicting healthy loans ("0"). Additionally, they also exhibited high recall for high-risk loans ("1"), indicating their ability to capture instances of high risk accurately. However, the Logistic Regression model trained on oversampled data achieved slightly better performance in terms of precision and recall for high-risk loans. Overall, both models had high accuracy and balanced performance in both classes. 

Since it's crucial to identify high-risk loans correctly, it is recommended to use the Logistic Regression model trained on oversampled data. This model maintains accurate predictions for healthy loans while striking a balance between precision and recall for potential high-risk loans. However, it's worth considering that model performance may depend on the specific problem and dataset characteristics. Therefore, it's recommended to continuously monitor and re-evaluate the model as new data becomes available or the problem context evolves. 

In conclusion, the oversampled data Logistic Regression model is recommended for predicting loan statuses due to its strong overall performance and balanced prediction capabilities for both healthy and high-risk loans.
