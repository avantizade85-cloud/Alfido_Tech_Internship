# Alfido Tech Internship – Project 2
# Loan Approval Prediction

## 📌 Project Overview

This project focuses on predicting whether a loan application will be approved or not using Machine Learning.

Two classification algorithms are used:

1. Logistic Regression
2. Random Forest

The models are evaluated using Precision, Recall, F1 Score and ROC-AUC.

SMOTE is used to handle class imbalance in the training dataset.

---

## 🎯 Project Objective

The main objectives of this project are:

- Analyze loan application data
- Clean and preprocess the dataset
- Handle missing values
- Encode categorical variables
- Scale numerical variables
- Handle class imbalance using SMOTE
- Train Machine Learning models
- Compare model performance
- Perform threshold analysis
- Identify important features
- Generate visualizations
- Provide business interpretation

---

## 📂 Project Structure

```text
Alfido_Loan_Approval/
│
├── data/
│   └── loan_dataset.csv
│
├── output/
│   ├── cleaned_loan_dataset.csv
│   ├── model_comparison.csv
│   ├── classification_report.txt
│   ├── threshold_analysis.csv
│   ├── feature_importance.csv
│   ├── business_interpretation.txt
│   └── final_project_summary.txt
│
├── Visualizations/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
│
├── loan_approval_prediction.py
└── README.md

## 🏁 Conclusion

This project successfully demonstrates the use of Machine Learning for loan approval prediction. The dataset was cleaned and preprocessed by handling missing values, encoding categorical features, scaling numerical features, and addressing class imbalance using SMOTE.

Logistic Regression and Random Forest models were trained and evaluated using Precision, Recall, F1 Score, and ROC-AUC. Model comparison and threshold analysis were performed to identify the better-performing model.

The project provides a complete Machine Learning workflow from data preprocessing to model evaluation and business interpretation. The developed model can be used as a decision-support tool for preliminary loan application screening.