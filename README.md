# 🛡️ CreditGuard AI – Credit Card Fraud Detection
*live demo*"https://creditguardai-7vzmxlywfswjl5ptem9x9e.streamlit.app/"

An intelligent Machine Learning-based system that detects potentially fraudulent credit card transactions using Python and Streamlit.

## 📌 Project Overview

CreditGuard AI is a machine learning project designed to identify suspicious credit card transactions. It analyzes transaction data, predicts fraud probabilities, classifies transaction risk levels and generates fraud alerts.

The system provides an interactive Streamlit dashboard for batch transaction analysis and model evaluation.

## 🎯 Objectives

- Detect potentially fraudulent credit card transactions.
- Apply machine learning techniques to transaction data.
- Analyze fraud probabilities and transaction risk.
- Reduce missed fraudulent transactions.
- Visualize model performance through an interactive dashboard.

## ✨ Key Features

- Credit card fraud prediction
- Fraud probability calculation
- Low, Medium and High risk classification
- Transaction analysis using CSV uploads
- Fraud alerts
- Confusion matrix visualization
- Accuracy, Precision, Recall and F1-score evaluation
- Downloadable fraud analysis reports

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| XGBoost | Classification model |
| Matplotlib | Data visualization |
| Streamlit | Web dashboard |
| Joblib | Model serialization |

## 📊 Dataset

**Credit Card Fraud Detection Dataset – Kaggle**

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- 284,807 transactions
- 492 fraudulent transactions
- 284,315 genuine transactions
- 30 input features and 1 target column

The target column is `Class`:
- 0 = Genuine transaction
- 1 = Fraudulent transaction

## ⚙️ Project Workflow

1. Dataset collection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis
4. Feature scaling
5. Handling class imbalance using SMOTE
6. Machine learning model training
7. Model evaluation
8. Fraud prediction
9. Streamlit dashboard development

## 📁 Project Structure

```text
CreditGuard-AI/
│
├── app.py
├── creditguard_model_v2.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/sonaliyadav3131-cmyk/CreditGuard-AI.git
```

### 2. Open the project folder

```bash
cd CreditGuard-AI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

### 5. Open in browser

```text
http://localhost:8501
```

Upload a compatible transaction CSV file to start fraud analysis.

## 📈 Model Evaluation

The project evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Precision-Recall AUC
- Confusion Matrix

*Final independent test-set metrics will be added after verification.*

## 🔮 Future Enhancements

- Real-time transaction monitoring
- Advanced anomaly detection
- Improved fraud detection models
- Interactive fraud analytics
- Cloud deployment


## 👩‍💻 Author

**Sonali Yadav**

Machine Learning Project | CreditGuard AI

---

⭐ If you find this project interesting, consider giving the repository a star!﻿

