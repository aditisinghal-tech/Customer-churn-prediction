# 📊 Telecom Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing information.

## 🎯 Project Objective

Customer churn can lead to significant revenue loss for telecom companies. The objective of this project is to build a classification model that identifies customers who are at higher risk of leaving, helping businesses take proactive retention actions.

## 📊 Dataset

The project uses a telecom customer churn dataset containing **7,043 customer records** and information related to:

- Customer demographics
- Tenure
- Phone and internet services
- Online security and support services
- Contract type
- Payment method
- Monthly and total charges
- Churn status

## 🔍 Exploratory Data Analysis

Key findings from the analysis:

- Customers with **lower tenure** show higher churn tendency.
- Customers with **higher monthly charges** are more likely to churn.
- **Month-to-month contracts** have substantially higher churn than long-term contracts.
- Customers without services such as **Online Security and Tech Support** show higher churn rates.
- **Electronic check** users have a higher churn rate compared with automatic payment methods.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Handled blank values in `TotalCharges`.
- Converted `TotalCharges` to numeric format.
- Removed duplicate records.
- Removed `customerID`.
- Encoded binary categorical variables.
- Applied one-hot encoding to multi-category variables.
- Split the data into training and testing sets.
- Applied `StandardScaler` for Logistic Regression.

## 🤖 Models Used

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Comparison

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.04% | 69% | 60% | 64% |
| Random Forest | 79.56% | 66% | 47% | 55% |
| XGBoost | 81.41% | 69% | 55% | 61% |

Logistic Regression provided the strongest baseline performance and was selected for the final application.

## 🎯 Final Model

The final application uses **Logistic Regression** with a classification threshold of **0.30**.

The threshold was lowered from the default 0.50 to prioritize identifying more potential churn customers.

### Final Performance

- **Accuracy:** 77.64%
- **Precision (Churn):** 55%
- **Recall (Churn):** 80%
- **F1-score (Churn):** 65%
- **ROC-AUC:** 86.21%
- **Classification Threshold:** 0.30

The higher recall helps identify a larger proportion of customers who actually churned.

## 🌐 Streamlit Application

The project includes an interactive Streamlit application where users can enter customer information and receive:

- Churn probability
- Churn risk level
- Final churn prediction

The application uses the saved trained model, scaler, feature names, and classification threshold.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Git & GitHub

## 🚀 How to Run Locally

Clone the repository:

```bash
git clone <your-repository-url>
