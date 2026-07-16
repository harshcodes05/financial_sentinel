# Financial Sentinel: Credit Card Fraud Detection using Machine Learning

An end-to-end machine learning project for detecting fraudulent credit card transactions using supervised learning. The project covers the complete workflow from data exploration and preprocessing to model training, evaluation, serialization, and deployment.

---

## Project Overview

Credit card fraud causes significant financial losses worldwide and presents a challenging machine learning problem due to the extreme class imbalance between legitimate and fraudulent transactions.

This project develops a fraud detection system capable of identifying fraudulent transactions while minimizing false alarms. The complete machine learning pipeline includes data quality assessment, exploratory data analysis, preprocessing, model development, evaluation, and deployment.

---

## Features

- Data Quality Assessment
- Exploratory Data Analysis (EDA)
- Class Imbalance Analysis
- Correlation Analysis
- Feature Scaling using StandardScaler
- SMOTE Oversampling
- Logistic Regression
- Random Forest Classifier
- Model Evaluation using multiple metrics
- ROC Curve Comparison
- Precision-Recall Curve Comparison
- Model Serialization using Joblib
- Streamlit Web Application

---

## Dataset

This project uses the **Credit Card Fraud Detection Dataset**.

Due to GitHub's file size limitations, the dataset is **not included** in this repository.

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the dataset inside the project directory:

```text
data/
└── creditcard.csv
```

---

## Project Structure

```text
financial_sentinel/
│
├── data/
│
├── models/
│   ├── random_forest_model.pkl
│   └── standard_scaler.pkl
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

---

## Machine Learning Pipeline

1. Data Loading
2. Data Quality Assessment
3. Exploratory Data Analysis
4. Feature Analysis
5. Data Preprocessing
6. Train-Test Split
7. Feature Scaling
8. SMOTE Oversampling
9. Model Training
10. Model Evaluation
11. Model Comparison
12. Model Serialization
13. Streamlit Deployment

---

## Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|---------:|----------:|--------:|---------:|---------:|
| Logistic Regression | 0.9741 | 0.0578 | 0.9184 | 0.1088 | 0.9708 |
| Random Forest | **0.9993** | **0.8256** | 0.7245 | **0.7717** | 0.9669 |

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Streamlit

---

## Installation

Clone the repository:

```bash
git clone https://github.com/harshcodes05/financial_sentinel.git
```

Move into the project directory:

```bash
cd financial_sentinel
```


Download the dataset from Kaggle and place it inside:

```text
data/
└── creditcard.csv
```

Run the Streamlit application:

```bash
https://financialsentinel-jhmuxkenwhtnototztp3ah.streamlit.app/
```

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Results

Two machine learning algorithms were evaluated for fraud detection.

- Logistic Regression achieved excellent recall (**91.84%**) but generated a high number of false positives, resulting in low precision.
- Random Forest achieved the best balance between precision, recall, and F1-score while maintaining excellent overall performance.

Based on these results, **Random Forest** was selected as the final model for deployment.

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Evaluation of XGBoost and LightGBM
- Model explainability using SHAP
- Threshold optimization for business-specific objectives
- Real-time fraud detection pipeline
- Cloud deployment

---

## Author

**Harsh Sharma**
